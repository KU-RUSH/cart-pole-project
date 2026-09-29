"""DQN 학습 루프: 행동 선택 -> 경험 저장 -> 신경망 업데이트."""

import math

from src.config import EnvConfig
from src.controllers.dqn import DEFAULT_CHECKPOINT, DQNAgent
from src.dqn_episode import run_dqn_episode
from src.environment import CartPoleEnv


# 학습을 시작하기 전에 이 값을 바꾸면 됩니다.
EPISODES = 300
OUTPUT_PATH = DEFAULT_CHECKPOINT


def train(episodes=300, output=DEFAULT_CHECKPOINT):
    if episodes < 1:
        raise ValueError("episodes는 1 이상이어야 합니다.")

    config = EnvConfig()
    # 학습률, gamma, 배치 크기 등을 바꾸려면 아래 생성자에 지정하세요.
    agent = DQNAgent(max_force=config.max_force)

    with CartPoleEnv(config) as env:
        for episode_index in range(episodes):
            # epsilon 확률로 무작위 행동을 고릅니다. 학습하면서 탐험을 줄입니다.
            epsilon = 0.05 + 0.95 * math.exp(-episode_index / 90.0)
            stats = run_dqn_episode(env, agent, epsilon)

            episode_number = episode_index + 1
            if episode_number == 1 or episode_number % 10 == 0:
                print(
                    f"episode={episode_number}  return={stats.total_reward:.2f}  "
                    f"steps={stats.steps}  epsilon={epsilon:.3f}  "
                    f"loss={stats.mean_loss:.4f}"
                )

    # 추론에 필요한 가중치만 저장합니다. 학습을 재개하는 체크포인트는 아닙니다.
    saved = agent.save(output)
    print(f"Model saved: {saved.resolve()}")
    return agent


def main():
    try:
        train(EPISODES, OUTPUT_PATH)
    except KeyboardInterrupt:
        print("학습을 중단했습니다.")
    except ValueError as error:
        raise SystemExit(f"오류: {error}") from error
    except NotImplementedError as error:
        raise SystemExit(f"DQN 스켈레톤이 아직 미완성입니다: {error}") from error


if __name__ == "__main__":
    main()
