"""DQN 학습 에피소드 한 번을 실행합니다."""

from dataclasses import dataclass

from src.controllers.dqn import DQNAgent
from src.environment import CartPoleEnv
from src.reward import calculate_reward


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    total_reward: float
    steps: int
    mean_loss: float


def run_dqn_episode(
    env: CartPoleEnv,
    agent: DQNAgent,
    epsilon: float,
) -> EpisodeStats:
    """행동 선택, 보상 계산, 경험 저장, 학습을 종료 시점까지 반복합니다."""
    state = env.reset()
    total_reward = 0.0
    losses: list[float] = []

    while True:
        action, force = agent.choose_action(state, epsilon)
        result = env.step(force)

        # 보상은 물리 환경이 아니라 reward.py에서 계산합니다.
        reward = calculate_reward(
            state,
            result.force,
            result.state,
            result.terminated,
            env.config,
        )
        agent.observe(state, action, reward, result.state, result.terminated)

        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        state = result.state
        total_reward += reward
        if result.done:
            break

    mean_loss = sum(losses) / len(losses) if losses else 0.0
    return EpisodeStats(total_reward, env.step_count, mean_loss)
