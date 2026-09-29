"""학생이 직접 수정할 파일입니다. compute()의 힘 계산부터 바꿔보세요.

main.py 위쪽의 CONTROLLER를 "my_controller"로 바꾼 뒤 실행하세요.
"""

import numpy as np

from src.state import State


class MyController:
    def __init__(self, control_hz: float, max_force: float) -> None:
        self.dt = 1.0 / control_hz  # 제어 주기 (초)
        self.max_force = max_force  # 힘의 최대 크기 (N)

    def reset(self) -> None:
        pass  # 적분값처럼 이전 스텝을 기억하는 변수가 생기면 여기서 초기화합니다.

    def compute(self, state: State) -> float:
        # 기본 예제: 막대의 각도·각속도와 카트의 위치·속도로 힘을 계산합니다.
        # 강화학습 정책에는 state.as_array()를 입력하면 됩니다.
        force = 75.0 * state.theta + 12.0 * state.theta_dot
        force += 4.0 * state.x + 5.0 * state.x_dot
        return float(np.clip(force, -self.max_force, self.max_force))


def make_controller(control_hz: float, max_force: float) -> MyController:
    """main.py가 호출합니다. 내가 만든 제어기 객체를 반환하세요."""
    return MyController(control_hz, max_force)
