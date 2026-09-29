from __future__ import annotations

import numpy as np

from src.state import State, wrap_angle


def linearized_cartpole() -> tuple[np.ndarray, np.ndarray]:
    """수직 근처에서 근사한 연속시간 모델: 상태 변화율 = A @ 상태 + B @ 힘.

    상태 순서는 [x, x_dot, theta, theta_dot]입니다.
    실제 캡슐 모양 막대를 균일한 막대로 근사하고 건마찰은 생략합니다.
    """
    cart_mass = 1.0
    pole_mass = 0.10
    pole_length = 0.60
    center_of_mass = pole_length / 2.0
    gravity = 9.81
    cart_damping = 0.10
    hinge_damping = 0.010
    pole_inertia = pole_mass * pole_length**2 / 3.0

    denominator = (
        (cart_mass + pole_mass) * pole_inertia
        - (pole_mass * center_of_mass) ** 2
    )
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [
                0.0,
                -pole_inertia * cart_damping / denominator,
                -(pole_mass * center_of_mass) ** 2 * gravity / denominator,
                pole_mass * center_of_mass * hinge_damping / denominator,
            ],
            [0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                pole_mass * center_of_mass * cart_damping / denominator,
                (cart_mass + pole_mass)
                * pole_mass
                * gravity
                * center_of_mass
                / denominator,
                -(cart_mass + pole_mass) * hinge_damping / denominator,
            ],
        ],
        dtype=np.float64,
    )
    B = np.array(
        [
            [0.0],
            [pole_inertia / denominator],
            [0.0],
            [-pole_mass * center_of_mass / denominator],
        ],
        dtype=np.float64,
    )
    return A, B


def solve_discrete_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    dt: float,
    iterations: int = 1_000,
) -> np.ndarray:
    """Q(상태 오차 비용)와 R(힘의 비용)로 피드백 이득 K를 계산합니다.

    연속시간 모델을 제어 주기 dt에 맞춰 근사하고 Riccati 식을 반복합니다.
    """
    A_d = np.eye(A.shape[0]) + A * dt
    B_d = B * dt
    P = Q.copy()

    for _ in range(iterations):
        gain = np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
        next_p = Q + A_d.T @ P @ (A_d - B_d @ gain)
        if np.max(np.abs(next_p - P)) < 1e-10:
            P = next_p
            break
        P = next_p

    return np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)


class LQRController:
    """상태 오차에 힘 u = -K @ 상태를 적용하는 제어기입니다."""

    def __init__(
        self,
        control_hz: float = 50.0,
        max_force: float = 10.0,
        q_weights: tuple[float, float, float, float] = (2.0, 0.5, 60.0, 3.0),
        r_weight: float = 0.5,
    ) -> None:
        if control_hz <= 0.0:
            raise ValueError("control_hz must be > 0")
        if max_force <= 0.0:
            raise ValueError("max_force must be > 0")

        A, B = linearized_cartpole()
        if len(q_weights) != 4 or not np.all(np.isfinite(q_weights)) or min(q_weights) < 0:
            raise ValueError("q_weights must contain four finite nonnegative values")
        if not np.isfinite(r_weight) or r_weight <= 0:
            raise ValueError("r_weight must be finite and > 0")
        Q = np.diag(q_weights)
        R = np.array([[r_weight]])
        self.gain = solve_discrete_lqr(A, B, Q, R, 1.0 / control_hz)
        self.max_force = float(max_force)

    def reset(self) -> None:
        pass  # LQR는 이전 스텝의 값을 저장하지 않습니다.

    def compute(self, state: State) -> float:
        error = np.array(
            [state.x, state.x_dot, wrap_angle(state.theta), state.theta_dot]
        )
        force = -float((self.gain @ error)[0])
        return float(np.clip(force, -self.max_force, self.max_force))


def make_controller(control_hz: float, max_force: float) -> LQRController:
    return LQRController(
        control_hz=control_hz,
        max_force=max_force,
        q_weights=(2.0, 0.5, 60.0, 3.0),  # 위치, 속도, 각도, 각속도의 비용
        r_weight=0.5,  # 힘을 크게 쓸 때의 비용
    )
