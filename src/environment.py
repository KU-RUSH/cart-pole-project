"""환경: 카트폴의 물리 계산과 종료 조건을 담당합니다.

보상은 학습 문제마다 달라지므로 reward.py에 따로 정의합니다.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

from src.config import EnvConfig
from src.state import State, wrap_angle


@dataclass(frozen=True, slots=True)
class StepResult:
    """시뮬레이션을 한 스텝 진행한 결과입니다."""

    state: State
    force: float
    disturbance: float
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class CartPoleEnv:
    """reset()은 초기 상태, step()은 다음 상태와 종료 여부를 반환합니다.

    force는 제어기가 만든 힘이고 disturbance는 카트에 가하는 외력입니다.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        path = Path(__file__).with_name("cartpole.xml")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        cart = self.model.joint("cart_slide")
        pole = self.model.joint("pole_hinge")
        self._cart_body_id = int(self.model.body("cart").id)
        self._qpos = [int(cart.qposadr[0]), int(pole.qposadr[0])]
        self._qvel = [int(cart.dofadr[0]), int(pole.dofadr[0])]
        actuator = self.model.actuator("cart_force")
        self._actuator_id = actuator.id
        if self.config.max_force > min(-actuator.ctrlrange[0], actuator.ctrlrange[1]):
            raise ValueError("max_force exceeds the model actuator limits (10 N)")
        if self.config.position_limit > min(-cart.range[0], cart.range[1]):
            raise ValueError("position_limit exceeds the model rail limits (2.4 m)")

        steps = 1.0 / (self.config.control_hz * self.model.opt.timestep)
        if steps < 1 or not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("control period must be an integer multiple of the physics timestep")
        self.physics_steps_per_action = round(steps)
        self.control_period = self.physics_steps_per_action * float(self.model.opt.timestep)
        # 실험 시간이 제어 주기로 나누어떨어지지 않으면 한 스텝 올림합니다.
        self.max_steps = max(1, math.ceil(self.config.duration_seconds / self.control_period - 1e-10))
        self.step_count = 0
        self.state = State()
        self._rng = np.random.default_rng()
        self._needs_reset = True
        self._viewer = None

    def _lock(self):
        return self._viewer.lock() if self._viewer is not None else nullcontext()

    def _read_state(self) -> State:
        x, theta = self.data.qpos[self._qpos]
        x_dot, theta_dot = self.data.qvel[self._qvel]
        return State(
            sim_time=float(self.data.time),
            x=float(x),
            x_dot=float(x_dot),
            theta=float(theta),
            theta_dot=float(theta_dot),
        )

    def reset(self, *, state: State | None = None) -> State:
        """수직 근처의 초기 상태에서 다시 시작합니다. 시간은 0으로 돌아갑니다."""
        if state is None:
            ranges = np.array([
                self.config.initial_position_range,
                self.config.initial_velocity_range,
                math.radians(self.config.initial_angle_degrees),
                self.config.initial_angular_velocity_range,
            ])
            x, x_dot, theta, theta_dot = self._rng.uniform(-ranges, ranges)
            state = State(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)
        if not np.all(np.isfinite(state.as_array())) or self._outside_limits(state):
            raise ValueError("initial state must be finite and inside the episode limits")
        with self._lock():
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self._qpos] = [state.x, state.theta]
            self.data.qvel[self._qvel] = [state.x_dot, state.theta_dot]
            mujoco.mj_forward(self.model, self.data)
            self.state = self._read_state()
        self.step_count = 0
        self._needs_reset = False
        return self.state

    def step(
        self,
        force: float,
        disturbance: float = 0.0,
        *,
        stop_on_failure: bool = True,
    ) -> StepResult:
        """제어력과 외란을 한 제어 주기 동안 적용합니다.

        disturbance는 액추에이터와 별개로 카트에 가하는 x축 외력(N)입니다.
        stop_on_failure=False이면 각도·위치 제한을 넘어도 계속 시뮬레이션합니다.
        """
        if self._needs_reset:
            raise RuntimeError("call reset() before stepping or after an episode ends")
        if np.ndim(force) != 0 or not np.isfinite(force):
            raise ValueError("force must be a finite scalar in Newtons")
        if np.ndim(disturbance) != 0 or not np.isfinite(disturbance):
            raise ValueError("disturbance must be a finite scalar in Newtons")
        applied_force = float(np.clip(force, -self.config.max_force, self.config.max_force))
        applied_disturbance = float(disturbance)
        terminated = False
        with self._lock():
            self.data.ctrl[self._actuator_id] = applied_force
            self.data.xfrc_applied[self._cart_body_id] = 0.0
            self.data.xfrc_applied[self._cart_body_id, 0] = applied_disturbance
            # 기본값은 제어 1스텝(0.02초) 동안 물리 계산 10번(각 0.002초)입니다.
            for _ in range(self.physics_steps_per_action):
                mujoco.mj_step(self.model, self.data)
                self.state = self._read_state()
                if not np.all(np.isfinite(self.state.as_array())):
                    self._needs_reset = True
                    raise FloatingPointError("MuJoCo produced a non-finite state")
                if stop_on_failure and self._outside_limits(self.state):
                    terminated = True
                    break
        self.step_count += 1
        # terminated: 쓰러짐/레일 이탈, truncated: 정해진 시간이 끝남.
        truncated = self.step_count >= self.max_steps and not terminated
        self._needs_reset = terminated or truncated
        return StepResult(
            self.state,
            applied_force,
            applied_disturbance,
            terminated,
            truncated,
        )

    def _outside_limits(self, state: State) -> bool:
        return (
            abs(state.x) >= self.config.position_limit
            or abs(wrap_angle(state.theta)) >= math.radians(self.config.angle_limit_degrees)
        )

    def render(self) -> bool:
        """현재 상태를 화면에 보여줍니다. 창을 닫으면 False를 반환합니다."""
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self._viewer.lock():
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._viewer.cam.fixedcamid = self.model.camera("side").id
        if not self._viewer.is_running():
            return False
        self._viewer.sync(state_only=True)
        return True

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self) -> "CartPoleEnv":
        return self

    def __exit__(self, *args) -> None:
        self.close()
