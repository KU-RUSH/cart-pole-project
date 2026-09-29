"""DQN 학습에 사용할 보상 함수.

학생이 주로 수정할 파일입니다. 환경은 물리 계산만 담당하고, 학습 목표는
이 함수에서 자유롭게 정합니다.
"""

import math

import numpy as np

from src.config import EnvConfig
from src.state import State, wrap_angle


def calculate_reward(
    state: State,
    force: float,
    next_state: State,
    terminated: bool,
    config: EnvConfig,
) -> float:
    """(상태, 행동, 다음 상태, 실패 여부)를 보고 보상을 반환합니다.

    아래 식은 시작용 예제입니다. 항목과 가중치, 실패 보상을 직접 바꿔보세요.
    """
    if terminated:
        return -1.0

    angle_cost = (
        wrap_angle(next_state.theta) / math.radians(config.angle_limit_degrees)
    ) ** 2
    position_cost = (next_state.x / config.position_limit) ** 2
    force_cost = (force / config.max_force) ** 2

    reward = 1.0 - 0.7 * angle_cost - 0.2 * position_cost - 0.1 * force_cost
    return float(np.clip(reward, 0.0, 1.0))
