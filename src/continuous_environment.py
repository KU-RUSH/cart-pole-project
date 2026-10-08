"""연속 힘 제어를 위한 표준 Gymnasium 인터페이스."""

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

from src.config import EnvConfig
from src.environment import CartPoleEnv
from src.reward import calculate_reward, reward_settings
from src.state import State


OBSERVATION_SCALE = np.array([.5, 1., math.radians(10), 2., 10.], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ContinuousControl:
    smoothing_alpha: float = .25
    max_force_change: float = 2.

    def __post_init__(self):
        if not math.isfinite(self.smoothing_alpha) or not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if not math.isfinite(self.max_force_change) or self.max_force_change <= 0:
            raise ValueError("max_force_change must be finite and > 0")

    def apply(self, state: State, action: float, max_force: float) -> tuple[float, float]:
        if not math.isfinite(action):
            raise ValueError("action must be finite")
        target = float(np.clip(action, -1, 1)) * max_force
        previous = float(np.clip(state.previous_force, -max_force, max_force))
        change = float(np.clip(self.smoothing_alpha * (target - previous),
                               -self.max_force_change, self.max_force_change))
        return target, float(np.clip(previous + change, -max_force, max_force))


def observe_state(state: State, max_force: float = 10., scale=None) -> np.ndarray:
    if scale is None:
        scale = OBSERVATION_SCALE.copy()
        scale[-1] = max_force
    values = np.append(state.as_array(), state.previous_force)
    return np.clip(values / scale, -2, 2).astype(np.float32)


class ContinuousCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config: EnvConfig | None = None, control: ContinuousControl | None = None,
                 render_mode: str | None = None, settings: dict | None = None):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human")
        self.config = config or EnvConfig(reward_profile="still")
        self.control = control or ContinuousControl()
        self.reward_settings = settings if settings is not None else reward_settings(self.config.reward_profile)
        self.observation_scale = OBSERVATION_SCALE.copy()
        self.observation_scale[-1] = self.config.max_force
        self.render_mode = render_mode
        self.physics = CartPoleEnv(self.config)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-2, 2, shape=(5,), dtype=np.float32)
        self.curriculum_fraction = 1.

    def set_curriculum_fraction(self, fraction: float):
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("curriculum fraction must be in (0, 1]")
        self.curriculum_fraction = float(fraction)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        initial = (options or {}).get("state")
        if initial is None:
            ranges = np.array([self.config.initial_position_range, self.config.initial_velocity_range,
                               math.radians(self.config.initial_angle_degrees),
                               self.config.initial_angular_velocity_range]) * self.curriculum_fraction
            x, velocity, angle, angular_velocity = self.np_random.uniform(-ranges, ranges)
            initial = State(x=x, x_dot=velocity, theta=angle, theta_dot=angular_velocity)
        state = self.physics.reset(seed=seed, state=initial)
        return observe_state(state, self.config.max_force, self.observation_scale), {}

    def step(self, action):
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (1,):
            raise ValueError("action must have shape (1,)")
        state = self.physics.state
        target, force = self.control.apply(state, float(values[0]), self.config.max_force)
        result = self.physics.step(force)
        reward = calculate_reward(state, result.force, result.state, result.terminated, self.config,
                                  self.reward_settings)
        info = {"target_force": target, "force": result.force}
        if result.done:
            info["is_success"] = result.truncated and not result.terminated
        return (observe_state(result.state, self.config.max_force, self.observation_scale), reward,
                result.terminated, result.truncated, info)

    def render(self):
        if self.render_mode == "human":
            return self.physics.render()

    def close(self):
        self.physics.close()
