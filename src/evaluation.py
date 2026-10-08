"""탐험이나 학습 없이 고정된 초기 상태들에서 모델을 평가합니다."""

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import EnvConfig
from src.environment import CartPoleEnv
from src.reward import calculate_reward
from src.state import State, wrap_angle


class ForcePolicy(Protocol):
    max_force: float
    reward_profile: str

    def choose_action(self, state: State) -> tuple[int | float, float]: ...
    def target_force_for_action(self, action: int | float) -> float: ...


@dataclass(frozen=True, slots=True)
class EvaluationStats:
    mean_reward: float
    mean_steps: float
    success_rate: float
    mean_abs_force: float
    mean_abs_force_change: float
    max_abs_force_change: float
    tail_position_rms: float
    tail_velocity_rms: float
    tail_angle_rms_degrees: float
    tail_angular_velocity_rms: float
    settled_rate: float


def is_settled(state) -> bool:
    """중앙 ±5cm, 속도 ±5cm/s, 각도 ±1도, 각속도 ±0.1rad/s."""
    return (abs(state.x) <= .05 and abs(state.x_dot) <= .05
            and abs(wrap_angle(state.theta)) <= np.deg2rad(1)
            and abs(state.theta_dot) <= .1)


def evaluate_network(
    network: ForcePolicy, config: EnvConfig, episodes: int = 10, seed: int = 10_000,
    trace_path: str | Path | None = None,
) -> EvaluationStats:
    if episodes < 1:
        raise ValueError("evaluation episodes must be >= 1")
    if not np.isclose(network.max_force, config.max_force):
        raise ValueError("checkpoint max_force differs from the environment")
    if network.reward_profile != config.reward_profile:
        raise ValueError("checkpoint reward_profile differs from the evaluation configuration")
    rewards, steps, successes = [], [], []
    forces, changes, trace = [], [], []
    tails, settled = [], []
    tail_start = max(0.0, config.duration_seconds - 2.0)
    # 학습 환경/RNG/replay에 영향을 주지 않는 별도의 평가 환경입니다.
    with CartPoleEnv(config, seed=seed) as env:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0.0
            episode_tail = []
            episode_settled = []
            while True:
                action, force = network.choose_action(state)
                result = env.step(force)
                reward = calculate_reward(state, result.force, result.state, result.terminated, config,
                                          getattr(network, "reward_settings", None))
                total_reward += reward
                forces.append(abs(result.force))
                changes.append(abs(result.force - state.previous_force))
                if result.state.sim_time >= tail_start - 1e-9:
                    episode_tail.append([result.state.x, result.state.x_dot,
                                         np.rad2deg(wrap_angle(result.state.theta)),
                                         result.state.theta_dot])
                    episode_settled.append(is_settled(result.state))
                if trace_path is not None:
                    trace.append([episode + 1, env.step_count, result.state.sim_time,
                                  result.state.x, result.state.theta, result.state.x_dot,
                                  result.state.theta_dot, network.target_force_for_action(action),
                                  result.force, reward, result.terminated, result.truncated])
                state = result.state
                if result.done:
                    rewards.append(total_reward)
                    steps.append(env.step_count)
                    successes.append(result.truncated and not result.terminated)
                    # 실패한 짧은 실행을 작은 RMS로 잘못 평가하지 않습니다.
                    survived = result.truncated and not result.terminated
                    if survived:
                        tails.extend(episode_tail)
                    settled.append(survived and bool(episode_settled) and all(episode_settled))
                    break
    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "step", "sim_time", "x", "theta", "x_dot", "theta_dot", "target_force",
                             "applied_force", "reward", "terminated", "truncated"])
            writer.writerows(trace)
    rms = np.sqrt(np.mean(np.square(tails), axis=0)) if tails else np.full(4, np.nan)
    return EvaluationStats(
        float(np.mean(rewards)), float(np.mean(steps)), float(np.mean(successes)),
        float(np.mean(forces)), float(np.mean(changes)), float(np.max(changes)),
        *map(float, rms), float(np.mean(settled)),
    )
