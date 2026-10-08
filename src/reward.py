"""강화학습에 사용할 보상 함수.

학생이 주로 수정할 파일입니다. 환경은 물리 계산만 담당하고, 학습 목표는
이 함수에서 자유롭게 정합니다.
"""

import math

import numpy as np

from src.config import EnvConfig
from src.state import State, wrap_angle


# 정지 목표의 오차 기준과 가중치입니다. 실패 경계와 분리해서 튜닝합니다.
# 기준값이 작을수록 같은 오차를 더 크게 벌줍니다.
STILL_POSITION_SCALE = 0.25  # m
STILL_ANGLE_SCALE_DEGREES = 5.0
STILL_VELOCITY_SCALE = 0.35  # m/s
STILL_ANGULAR_VELOCITY_SCALE = 1.0  # rad/s
STILL_WEIGHTS = {"position": 0.6, "angle": 1.0, "velocity": 0.4,
                 "angular_velocity": 0.2, "force": 0.02, "force_change": 0.02}
REGULATION_WEIGHTS = {"position": 20., "velocity": 5., "angle": 100.,
                      "angular_velocity": 5., "force": .001, "force_change": .005}
REGULATION_FAILURE_COST = 1000.


def reward_settings(profile: str) -> dict:
    """실험 시작 시 보상 상수를 복사해 체크포인트와 작업자에 전달합니다."""
    if profile == "regulate":
        return {"weights": dict(REGULATION_WEIGHTS), "failure_cost": REGULATION_FAILURE_COST}
    if profile == "still":
        return {"weights": dict(STILL_WEIGHTS), "position_scale": STILL_POSITION_SCALE,
                "angle_scale_degrees": STILL_ANGLE_SCALE_DEGREES,
                "velocity_scale": STILL_VELOCITY_SCALE,
                "angular_velocity_scale": STILL_ANGULAR_VELOCITY_SCALE}
    return {}


def calculate_reward(
    state: State,
    force: float,
    next_state: State,
    terminated: bool,
    config: EnvConfig,
    settings: dict | None = None,
) -> float:
    """(상태, 행동, 다음 상태, 실패 여부)를 보고 보상을 반환합니다.

    아래 식은 시작용 예제입니다. 항목과 가중치, 실패 보상을 직접 바꿔보세요.
    """
    settings = settings or reward_settings(config.reward_profile)
    if terminated:
        if config.reward_profile == "regulate":
            return -settings["failure_cost"]
        return -1.0

    if config.reward_profile == "regulate":
        weights = settings["weights"]
        # 정지점의 최댓값이 0인 이차 비용. 먼 상태에서도 오차 차이를 구분합니다.
        cost = (
            weights["position"] * next_state.x ** 2
            + weights["velocity"] * next_state.x_dot ** 2
            + weights["angle"] * wrap_angle(next_state.theta) ** 2
            + weights["angular_velocity"] * next_state.theta_dot ** 2
            + weights["force"] * force ** 2
            + weights["force_change"] * (force - state.previous_force) ** 2
        )
        return -float(cost)

    if config.reward_profile == "still":
        weights = settings["weights"]
        cost = (
            weights["position"] * (next_state.x / settings["position_scale"]) ** 2
            + weights["angle"] * (wrap_angle(next_state.theta) / math.radians(settings["angle_scale_degrees"])) ** 2
            + weights["velocity"] * (next_state.x_dot / settings["velocity_scale"]) ** 2
            + weights["angular_velocity"] * (next_state.theta_dot / settings["angular_velocity_scale"]) ** 2
            + weights["force"] * (force / config.max_force) ** 2
            + weights["force_change"] * ((force - state.previous_force) / 2.0) ** 2
        )
        # 목표에서 멀어도 생존 점수를 주고, 정지에 가까울수록 추가 점수를 줍니다.
        # 비용이 커져도 clip으로 넓은 0점 영역을 만들지 않습니다.
        return 0.2 + 0.8 * math.exp(-cost)

    angle_cost = (
        wrap_angle(next_state.theta) / math.radians(config.angle_limit_degrees)
    ) ** 2
    position_cost = (next_state.x / config.position_limit) ** 2
    force_cost = (force / config.max_force) ** 2
    force_change_cost = ((force - state.previous_force) / config.max_force) ** 2

    reward = (1.0 - 0.7 * angle_cost - 0.2 * position_cost
              - 0.1 * force_cost - 0.05 * force_change_cost)
    return float(np.clip(reward, 0.0, 1.0))
