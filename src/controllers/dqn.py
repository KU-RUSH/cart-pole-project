"""직접 완성하는 DQN 스켈레톤.

학습 순서는 ``train.py``에서 확인하고, 이 파일의 TODO를 위에서부터
구현하세요. PID/LQR과 달리 이 파일은 의도적으로 완성되어 있지 않습니다.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from src.state import State


STATE_SIZE = 4
ACTION_SIZE = 2
STATE_SCALE = np.array([2.35, 3.0, np.deg2rad(25.0), 5.0], dtype=np.float32)
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "results" / "dqn.npz"


def normalize_state(state: State) -> np.ndarray:
    """상태값의 크기를 신경망이 다루기 좋은 범위로 맞춥니다."""
    return np.clip(state.as_array() / STATE_SCALE, -2.0, 2.0).astype(np.float32)


class ReplayBuffer:
    """최근 경험을 저장했다가 무작위 미니배치로 꺼내는 저장소."""

    def __init__(self, capacity: int = 50_000) -> None:
        self._items: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self._items.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = np.random.default_rng().choice(
            len(self._items), size=batch_size, replace=False
        )
        batch = [self._items[index] for index in indices]
        states, actions, rewards, next_states, terminated = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(terminated, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._items)


class QNetwork:
    """상태를 받아 왼쪽/오른쪽 행동의 Q값을 출력할 모델 자리."""

    def __init__(self) -> None:
        # TODO 1-a: 사용할 신경망과 optimizer를 만드세요.
        # PyTorch를 쓴다면 nn.Module을 이 클래스 안에 두어도 좋습니다.
        pass

    def predict(self, states: np.ndarray) -> np.ndarray:
        """shape=(batch, 4)를 받아 shape=(batch, 2)의 Q값을 반환합니다."""
        # TODO 1-b: 신경망의 순전파를 구현하세요.
        raise NotImplementedError("QNetwork.predict()를 구현하세요")

    def update(
        self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray
    ) -> float:
        """선택한 행동의 Q값을 targets에 가깝게 학습하고 loss를 반환합니다."""
        # TODO 1-c: loss 계산, 역전파, optimizer step을 구현하세요.
        raise NotImplementedError("QNetwork.update()를 구현하세요")

    def copy_from(self, other: "QNetwork") -> None:
        """target network에 online network의 가중치를 복사합니다."""
        # TODO 1-d: 신경망 가중치를 복사하세요.
        raise NotImplementedError("QNetwork.copy_from()을 구현하세요")

    def save(self, path: str | Path) -> Path:
        # TODO 1-e: 신경망 가중치를 파일에 저장하세요.
        raise NotImplementedError("QNetwork.save()를 구현하세요")

    @classmethod
    def load(cls, path: str | Path) -> "QNetwork":
        # TODO 1-f: 저장한 가중치로 신경망을 복원하세요.
        raise NotImplementedError("QNetwork.load()를 구현하세요")


class DQNAgent:
    """epsilon-greedy 행동 선택과 DQN 업데이트를 연결하는 학습 에이전트."""

    def __init__(
        self,
        max_force: float,
        gamma: float = 0.99,
        batch_size: int = 64,
        warmup_steps: int = 500,
        target_update_steps: int = 250,
    ) -> None:
        self.actions = np.array([-max_force, max_force], dtype=np.float32)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_steps = int(target_update_steps)
        self.online = QNetwork()
        self.target = QNetwork()
        self.target.copy_from(self.online)
        self.replay = ReplayBuffer()
        self.environment_steps = 0

    def choose_action(self, state: State, epsilon: float) -> tuple[int, float]:
        """epsilon 확률로 탐험하고, 아니면 Q값이 가장 큰 행동을 고릅니다."""
        # TODO 2: epsilon-greedy 행동 선택을 구현하세요.
        raise NotImplementedError("DQNAgent.choose_action()을 구현하세요")

    def observe(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        terminated: bool,
    ) -> None:
        self.replay.add(
            normalize_state(state),
            action,
            reward,
            normalize_state(next_state),
            terminated,
        )
        self.environment_steps += 1

    def learn(self) -> float | None:
        """replay에서 한 배치를 뽑아 online network를 한 번 업데이트합니다."""
        if len(self.replay) < max(self.warmup_steps, self.batch_size):
            return None

        # TODO 3:
        # 1) replay.sample()로 배치를 꺼냅니다.
        # 2) target network로 다음 상태의 최대 Q값을 구합니다.
        # 3) target = reward + gamma * (1 - terminated) * next_q를 만듭니다.
        # 4) online.update()를 호출합니다.
        # 5) 일정 주기마다 target.copy_from(online)을 호출합니다.
        raise NotImplementedError("DQNAgent.learn()을 구현하세요")

    def save(self, path: str | Path) -> Path:
        return self.online.save(path)


class DQNController:
    """학습한 Q-network를 main.py 실행 형식에 연결하는 제어기."""

    def __init__(self, model_path: str | Path, max_force: float) -> None:
        self.network = QNetwork.load(model_path)
        self.actions = np.array([-max_force, max_force], dtype=np.float32)

    def reset(self) -> None:
        pass

    def compute(self, state: State) -> float:
        q_values = self.network.predict(normalize_state(state)[None, :])[0]
        return float(self.actions[int(np.argmax(q_values))])


def make_controller(control_hz: float, max_force: float) -> DQNController:
    """TODO를 완성한 뒤 main.py의 CONTROLLER를 "dqn"으로 설정하면 호출됩니다."""
    return DQNController(DEFAULT_CHECKPOINT, max_force=max_force)
