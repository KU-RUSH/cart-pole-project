from __future__ import annotations

import numpy as np

from src.state import State, wrap_angle


class PIDController:
    """막대 각도에 PID를 적용하고, 카트가 중앙으로 돌아오도록 힘을 더합니다.

    kp: 현재 기울기, ki: 기울기의 누적값, kd: 기울어지는 속도에 대한 이득.
    먼저 ki=0으로 두고 P와 D의 역할을 비교해보세요.
    """

    def __init__(
        self,
        control_hz: float = 50.0,
        max_force: float = 10.0,
        kp: float = 75.0,
        ki: float = 2.0,
        kd: float = 12.0,
        position_kp: float = 4.0,
        position_kd: float = 5.0,
        integral_limit: float = 0.25,
    ) -> None:
        if control_hz <= 0.0:
            raise ValueError("control_hz must be > 0")
        if max_force <= 0.0:
            raise ValueError("max_force must be > 0")

        self.dt = 1.0 / float(control_hz)
        self.max_force = float(max_force)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.position_kp = float(position_kp)
        self.position_kd = float(position_kd)
        self.integral_limit = float(integral_limit)
        self.reset()

    def reset(self) -> None:
        # 새 실험을 시작할 때 이전 실험의 적분값을 버립니다.
        self._angle_integral = 0.0

    def compute(self, state: State) -> float:
        angle = wrap_angle(state.theta)
        # 적분 = 매 스텝의 각도 × 제어 주기를 누적한 값.
        self._angle_integral += angle * self.dt
        self._angle_integral = float(
            np.clip(
                self._angle_integral,
                -self.integral_limit,
                self.integral_limit,
            )
        )

        pole_force = (
            self.kp * angle
            + self.ki * self._angle_integral
            + self.kd * state.theta_dot
        )
        # 카트를 잠깐 더 오른쪽으로 밀면 막대는 왼쪽으로 기울어집니다.
        # 이어서 각도 제어가 둘을 중앙 쪽으로 되돌리므로 위치 항의 부호가 +입니다.
        centering_force = (
            self.position_kp * state.x + self.position_kd * state.x_dot
        )
        force = pole_force + centering_force
        return float(np.clip(force, -self.max_force, self.max_force))


def make_controller(control_hz: float, max_force: float) -> PIDController:
    # 실습할 때 kp, ki, kd 등을 여기에 지정해 바꿔보세요.
    return PIDController(control_hz=control_hz, max_force=max_force)
