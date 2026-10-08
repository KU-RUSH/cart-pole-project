"""물리 환경 설정. 새 학습에 적용하며, 모델 실행은 저장된 설정을 복원합니다."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, slots=True)
class EnvConfig:
    control_hz: float = 50.0  # 1초에 힘을 계산하는 횟수
    max_force: float = 10.0  # 최대 힘 (N), 모델의 한계는 10 N
    duration_seconds: float = 10.0  # 한 번의 실험 길이 (초)
    position_limit: float = 2.35  # 이 위치를 벗어나면 실패 (m)
    angle_limit_degrees: float = 25.0  # 이 각도를 벗어나면 실패 (도)

    # 초기 상태를 정할 때 사용하는 범위: -값 ~ +값
    initial_position_range: float = 0.20
    initial_velocity_range: float = 0.15
    initial_angle_degrees: float = 8.0
    initial_angular_velocity_range: float = 0.20
    reward_profile: str = "balance"  # balance / still / regulate(이차 상태 비용)

    def __post_init__(self) -> None:
        # 객체를 만들 때 잘못된 설정을 확인합니다. 실습은 위의 값부터 바꾸세요.
        for field in fields(self):
            if field.name == "reward_profile":
                if self.reward_profile not in ("balance", "still", "regulate"):
                    raise ValueError("reward_profile must be balance, still or regulate")
                continue
            value = getattr(self, field.name)
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if field.name.startswith("initial_"):
                if value < 0.0:
                    raise ValueError(f"{field.name} must be >= 0")
            elif value <= 0.0:
                raise ValueError(f"{field.name} must be > 0")
        if self.angle_limit_degrees > 180.0:
            raise ValueError("angle_limit_degrees must be <= 180")
        if self.initial_position_range >= self.position_limit:
            raise ValueError("initial_position_range must be below position_limit")
        if self.initial_angle_degrees >= self.angle_limit_degrees:
            raise ValueError("initial_angle_degrees must be below angle_limit_degrees")
