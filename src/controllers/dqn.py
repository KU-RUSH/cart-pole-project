"""NumPy로 구현한 Double DQN: replay, target network, Huber loss, Adam."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.state import State


STATE_SIZE = 5
ACTION_SIZE = 11
STATE_SCALE = np.array([2.35, 3.0, np.deg2rad(25.0), 5.0, 10.0], dtype=np.float32)
STILL_STATE_SCALE = np.array([0.5, 1.0, np.deg2rad(10.0), 2.0, 10.0], dtype=np.float32)
STILL_ACTION_FRACTIONS = np.array([-1, -.5, -.2, -.075, -.025, 0, .025, .075, .2, .5, 1], dtype=np.float32)


def normalize_state(state: State, scale: np.ndarray = STATE_SCALE) -> np.ndarray:
    """물리 상태 + 이전 실제 힘. 기존 4입력 모델도 동일한 API로 복원합니다."""
    values = state.as_array()
    if len(scale) == 5:
        values = np.append(values, state.previous_force)
    return np.clip(values / scale, -2.0, 2.0).astype(np.float32)


class ReplayBuffer:
    """고정 크기 배열을 사용하는 순환 경험 저장소."""

    def __init__(self, capacity: int = 50_000, seed: int | None = None,
                 state_size: int = STATE_SIZE) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.states = np.empty((capacity, state_size), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self._size = 0
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminated) -> None:
        index = self._cursor
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminated[index] = terminated
        self._cursor = (index + 1) % len(self.actions)
        self._size = min(self._size + 1, len(self.actions))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return (
            self.states[indices], self.actions[indices], self.rewards[indices],
            self.next_states[indices], self.terminated[indices],
        )

    def __len__(self) -> int:
        return self._size


class QNetwork:
    """5 → 64 → 64 → 11 MLP. 목표 힘을 선택하고 이전 힘에서 부드럽게 이동합니다."""

    def __init__(self, learning_rate: float = 1e-3, seed: int | None = None, *,
                 input_size: int = STATE_SIZE, action_size: int = ACTION_SIZE,
                 max_force: float = 10.0, smoothing_alpha: float = 0.25,
                 max_force_change: float = 2.0, reward_profile: str = "balance") -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0")
        if input_size not in (4, 5) or action_size < 2:
            raise ValueError("input_size must be 4 or 5 and action_size must be >= 2")
        if not np.isfinite(max_force) or max_force <= 0:
            raise ValueError("max_force must be finite and > 0")
        if not np.isfinite(smoothing_alpha) or not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if not np.isfinite(max_force_change) or max_force_change <= 0:
            raise ValueError("max_force_change must be finite and > 0")
        if input_size == 4 and (smoothing_alpha != 1 or max_force_change < 2 * max_force):
            raise ValueError("smoothed policies need previous force as a fifth input")
        if reward_profile not in ("balance", "still"):
            raise ValueError("reward_profile must be balance or still")
        if reward_profile == "still" and (input_size != 5 or action_size != ACTION_SIZE):
            raise ValueError("still policies require 5 inputs and 11 actions")
        rng = np.random.default_rng(seed)
        sizes = (input_size, 64, 64, action_size)
        self.params: dict[str, np.ndarray] = {}
        for layer, (inputs, outputs) in enumerate(zip(sizes, sizes[1:]), start=1):
            self.params[f"W{layer}"] = (
                rng.normal(size=(inputs, outputs)) * np.sqrt(2.0 / inputs)
            ).astype(np.float32)
            self.params[f"b{layer}"] = np.zeros(outputs, dtype=np.float32)
        self.learning_rate = learning_rate
        self.state_scale = STATE_SCALE[:input_size].copy()
        if input_size == 5:
            self.state_scale[-1] = max_force
        self.max_force = float(max_force)
        self.action_forces = np.linspace(-max_force, max_force, action_size, dtype=np.float32)
        self.reward_profile = reward_profile
        if reward_profile == "still":
            self.state_scale = STILL_STATE_SCALE.copy()
            self.state_scale[-1] = max_force
            self.action_forces = STILL_ACTION_FRACTIONS * max_force
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_force_change = float(max_force_change)
        self._moments = {key: np.zeros_like(value) for key, value in self.params.items()}
        self._variances = {key: np.zeros_like(value) for key, value in self.params.items()}
        self._updates = 0

    def _forward(self, states: np.ndarray):
        inputs = np.asarray(states, dtype=np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != len(self.state_scale):
            raise ValueError(f"states must have shape (batch, {len(self.state_scale)})")
        hidden1 = np.maximum(inputs @ self.params["W1"] + self.params["b1"], 0.0)
        hidden2 = np.maximum(hidden1 @ self.params["W2"] + self.params["b2"], 0.0)
        q_values = hidden2 @ self.params["W3"] + self.params["b3"]
        return inputs, hidden1, hidden2, q_values

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._forward(states)[-1]

    def force_for_action(self, state: State, action: int) -> float:
        """학습·평가·화면 실행이 공유하는 힘 변환. 마지막 실제 힘에서 시작합니다."""
        previous = float(np.clip(state.previous_force, -self.max_force, self.max_force))
        target = float(self.action_forces[action])
        change = float(np.clip(self.smoothing_alpha * (target - previous),
                               -self.max_force_change, self.max_force_change))
        return float(np.clip(previous + change, -self.max_force, self.max_force))

    def choose_action(self, state: State) -> tuple[int, float]:
        inputs = normalize_state(state, self.state_scale)[None, :]
        action = int(np.argmax(self.predict(inputs)[0]))
        return action, self.force_for_action(state, action)

    def target_force_for_action(self, action: int) -> float:
        return float(self.action_forces[action])

    def _loss_and_gradients(self, states, actions, targets):
        inputs, hidden1, hidden2, q_values = self._forward(states)
        rows = np.arange(len(inputs))
        errors = q_values[rows, actions] - targets
        absolute = np.abs(errors)
        # Huber loss: 오차가 큰 경험이 학습을 과도하게 흔들지 않게 합니다.
        loss = float(np.mean(np.where(absolute <= 1.0, 0.5 * errors**2, absolute - 0.5)))
        output_gradient = np.zeros_like(q_values)
        output_gradient[rows, actions] = np.clip(errors, -1.0, 1.0) / len(inputs)
        gradients = {
            "W3": hidden2.T @ output_gradient,
            "b3": output_gradient.sum(axis=0),
        }
        gradient2 = (output_gradient @ self.params["W3"].T) * (hidden2 > 0)
        gradients.update(W2=hidden1.T @ gradient2, b2=gradient2.sum(axis=0))
        gradient1 = (gradient2 @ self.params["W2"].T) * (hidden1 > 0)
        gradients.update(W1=inputs.T @ gradient1, b1=gradient1.sum(axis=0))
        return loss, gradients

    def update(self, states, actions, targets) -> float:
        loss, gradients = self._loss_and_gradients(states, actions, targets)
        norm = np.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in gradients.values()))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise FloatingPointError("DQN loss/gradient is not finite")
        scale = min(1.0, 10.0 / max(norm, 1e-12))
        self._updates += 1
        for key, parameter in self.params.items():
            gradient = gradients[key] * scale
            moment, variance = self._moments[key], self._variances[key]
            moment *= 0.9
            moment += 0.1 * gradient
            variance *= 0.999
            variance += 0.001 * gradient**2
            corrected_moment = moment / (1.0 - 0.9**self._updates)
            corrected_variance = variance / (1.0 - 0.999**self._updates)
            parameter -= self.learning_rate * corrected_moment / (np.sqrt(corrected_variance) + 1e-8)
        return loss

    def copy_from(self, other: "QNetwork") -> None:
        for key in self.params:
            np.copyto(self.params[key], other.params[key])
        self.state_scale = other.state_scale.copy()
        self.action_forces = other.action_forces.copy()
        self.max_force = other.max_force
        self.smoothing_alpha = other.smoothing_alpha
        self.max_force_change = other.max_force_change
        self.reward_profile = other.reward_profile

    def save(self, path: str | Path, *, max_force: float | None = None) -> Path:
        """추론용 가중치와 입력/행동 설정을 원자적으로 저장합니다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        metadata = {
            "state_scale": self.state_scale, "format_version": np.array(3),
            "reward_profile": np.array(self.reward_profile),
            "action_forces": self.action_forces,
            "smoothing_alpha": np.array(self.smoothing_alpha),
            "max_force_change": np.array(self.max_force_change),
        }
        force = max_force if max_force is not None else self.max_force
        if force is not None:
            metadata["max_force"] = np.array(force)
        with temporary.open("wb") as handle:
            np.savez(handle, **self.params, **metadata)
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QNetwork":
        with np.load(path, allow_pickle=False) as checkpoint:
            if "W1" not in checkpoint or "W3" not in checkpoint:
                raise ValueError("checkpoint is missing W1/W3")
            first, last = checkpoint["W1"], checkpoint["W3"]
            if first.ndim != 2 or last.ndim != 2:
                raise ValueError("invalid checkpoint weight dimensions")
            max_force = float(checkpoint["max_force"]) if "max_force" in checkpoint else 10.0
            modern = "action_forces" in checkpoint
            if modern and ("smoothing_alpha" not in checkpoint or "max_force_change" not in checkpoint):
                raise ValueError("checkpoint is missing force smoothing settings")
            if not modern and (first.shape[0] != 4 or last.shape[1] != 2):
                raise ValueError("checkpoint needs action/force metadata")
            network = cls(
                input_size=first.shape[0], action_size=last.shape[1], max_force=max_force,
                smoothing_alpha=float(checkpoint["smoothing_alpha"]) if modern else 1.0,
                max_force_change=float(checkpoint["max_force_change"]) if modern else 2 * max_force,
                reward_profile=str(checkpoint["reward_profile"]) if "reward_profile" in checkpoint else "balance",
            )
            for key, parameter in network.params.items():
                if key not in checkpoint:
                    raise ValueError(f"checkpoint is missing {key}")
                value = np.asarray(checkpoint[key], dtype=np.float32)
                if value.shape != parameter.shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"invalid checkpoint parameter: {key}")
                np.copyto(parameter, value)
            if "state_scale" in checkpoint:
                scale = np.asarray(checkpoint["state_scale"], dtype=np.float32)
                if scale.shape != network.state_scale.shape or not np.all(np.isfinite(scale) & (scale > 0)):
                    raise ValueError("invalid checkpoint state_scale")
                network.state_scale = scale.copy()
            if modern:
                actions = np.asarray(checkpoint["action_forces"], dtype=np.float32)
                if (actions.shape != network.action_forces.shape
                        or not np.all(np.isfinite(actions))
                        or not np.all(np.diff(actions) > 0)
                        or np.max(np.abs(actions)) > max_force + 1e-6):
                    raise ValueError("invalid checkpoint action_forces")
                network.action_forces = actions.copy()
        return network


class DQNAgent:
    """환경 경험을 모아 하나의 online/target network를 학습합니다."""

    def __init__(
        self, max_force: float, gamma: float = 0.99, batch_size: int = 64,
        warmup_steps: int = 1_000, target_update_steps: int = 250,
        learning_rate: float = 1e-3, train_frequency: int = 4,
        replay_capacity: int = 50_000, seed: int | None = None,
        smoothing_alpha: float = 0.25, max_force_change: float = 2.0,
        reward_profile: str = "balance",
    ) -> None:
        if not np.isfinite(max_force) or max_force <= 0:
            raise ValueError("max_force must be finite and > 0")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if min(batch_size, warmup_steps, target_update_steps, train_frequency) < 1:
            raise ValueError("DQN step counts and batch_size must be >= 1")
        if replay_capacity < max(batch_size, warmup_steps):
            raise ValueError("replay_capacity must cover batch_size and warmup_steps")
        self.gamma = float(gamma)
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_steps = target_update_steps
        self.train_frequency = train_frequency
        seeds = np.random.SeedSequence(seed).generate_state(3)
        self._rng = np.random.default_rng(int(seeds[0]))
        self.online = QNetwork(learning_rate, seed=int(seeds[1]), max_force=max_force,
                               smoothing_alpha=smoothing_alpha, max_force_change=max_force_change,
                               reward_profile=reward_profile)
        self.actions = self.online.action_forces
        self.target = QNetwork(learning_rate, seed=0)
        self.target.copy_from(self.online)
        self.replay = ReplayBuffer(replay_capacity, seed=int(seeds[2]))
        self.environment_steps = 0
        self.learning_steps = 0
        self._next_update = max(warmup_steps, batch_size)

    def choose_actions(self, states: list[State], epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """Q값은 배치로 계산하고 환경별로 독립적으로 탐험합니다."""
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        inputs = np.stack([normalize_state(state, self.online.state_scale) for state in states])
        actions = np.argmax(self.online.predict(inputs), axis=1)
        explore = self._rng.random(len(states)) < epsilon
        actions[explore] = self._rng.integers(len(self.actions), size=int(explore.sum()))
        forces = np.array([self.online.force_for_action(state, int(action))
                           for state, action in zip(states, actions)], dtype=np.float64)
        return actions, forces

    def choose_action(self, state: State, epsilon: float) -> tuple[int, float]:
        actions, forces = self.choose_actions([state], epsilon)
        return int(actions[0]), float(forces[0])

    def observe(self, state, action, reward, next_state, terminated) -> None:
        inputs = normalize_state(state, self.online.state_scale)
        next_inputs = normalize_state(next_state, self.online.state_scale)
        self.replay.add(inputs, action, reward, next_inputs, terminated)
        self.environment_steps += 1

    def learn(self) -> float | None:
        """경험 4개당 한 번 학습. None까지 호출하면 밀린 업데이트를 처리합니다."""
        if self.environment_steps < self._next_update:
            return None
        states, actions, rewards, next_states, terminated = self.replay.sample(self.batch_size)
        # Double DQN: 행동 선택은 online, 그 행동의 평가는 target이 담당합니다.
        next_actions = np.argmax(self.online.predict(next_states), axis=1)
        next_q = self.target.predict(next_states)[np.arange(self.batch_size), next_actions]
        targets = rewards + self.gamma * (1.0 - terminated) * next_q
        loss = self.online.update(states, actions, targets)
        self.learning_steps += 1
        self._next_update += self.train_frequency
        if self.learning_steps % self.target_update_steps == 0:
            self.target.copy_from(self.online)
        return loss

    def save(self, path: str | Path) -> Path:
        return self.online.save(path)


class DQNController:
    """학습한 모델을 main.py의 제어기 인터페이스에 연결합니다."""

    def __init__(self, model_path: str | Path, max_force: float) -> None:
        self.network = QNetwork.load(model_path)
        if self.network.max_force is not None and not np.isclose(self.network.max_force, max_force):
            raise ValueError("checkpoint max_force differs from the environment")
        self.actions = self.network.action_forces

    def reset(self) -> None:
        pass

    def compute(self, state: State) -> float:
        return self.network.choose_action(state)[1]
