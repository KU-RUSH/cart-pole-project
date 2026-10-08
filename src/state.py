from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class State:
    """제어기가 받는 상태: 위치(m), 속도(m/s), 각도(rad), 각속도(rad/s).

    theta=0은 수직 위쪽이고, 양수이면 오른쪽으로 기울어진 상태입니다.
    """

    sim_time: float = 0.0
    x: float = 0.0
    x_dot: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0
    previous_force: float = 0.0  # 이 상태에 도달할 때 실제 적용한 힘(N); reset 시 0

    def as_array(self) -> np.ndarray:
        """강화학습 입력용 배열: [x, x_dot, theta, theta_dot]."""
        return np.array(
            [self.x, self.x_dot, self.theta, self.theta_dot],
            dtype=np.float64,
        )


def wrap_angle(angle: float) -> float:
    """각도를 -pi ~ pi 범위로 바꿉니다."""
    return math.atan2(math.sin(angle), math.cos(angle))
