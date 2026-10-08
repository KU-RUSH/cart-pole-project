"""여러 MuJoCo 환경에서 경험을 수집합니다. Windows에서도 spawn으로 실행합니다."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback

from src.config import EnvConfig
from src.environment import CartPoleEnv, StepResult
from src.reward import calculate_reward
from src.state import State


@dataclass(frozen=True, slots=True)
class VectorStep:
    result: StepResult        # 종료 직전 상태: replay에 저장할 next_state
    reward: float
    next_state: State         # 종료 시 자동 reset한 상태: 다음 행동의 입력
    episode_steps: int


def _advance(env: CartPoleEnv, force: float) -> VectorStep:
    state = env.state
    result = env.step(force)
    reward = calculate_reward(state, result.force, result.state, result.terminated, env.config)
    steps = env.step_count
    next_state = env.reset() if result.done else result.state
    return VectorStep(result, reward, next_state, steps)


def _environment_worker(connection: Connection, config: EnvConfig, seed: int) -> None:
    """프로세스마다 독립된 MjModel/MjData를 소유합니다. 신경망은 부모에만 있습니다."""
    try:
        with CartPoleEnv(config, seed=seed) as env:
            connection.send(("ok", env.reset()))
            while True:
                command, payload = connection.recv()
                if command == "close":
                    break
                if command != "step":
                    raise ValueError(f"unknown worker command: {command}")
                connection.send(("ok", _advance(env, payload)))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class VectorCartPoleEnv:
    """num_envs=1은 현재 프로세스, 2 이상은 환경별 자식 프로세스에서 실행합니다."""

    def __init__(self, config: EnvConfig, num_envs: int = 4, seed: int = 42) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._local: CartPoleEnv | None = None
        self._closed = False
        try:
            if num_envs == 1:
                self._local = CartPoleEnv(config, seed=seed)
                self.states = [self._local.reset()]
            else:
                context = mp.get_context("spawn")
                for index in range(num_envs):
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=_environment_worker, args=(child, config, seed + index),
                        name=f"cartpole-env-{index}", daemon=True,
                    )
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        child.close()
                        raise
                    child.close()
                    self._connections.append(parent)
                    self._processes.append(process)
                self.states = [self._receive(index) for index in range(num_envs)]
        except BaseException:
            self.close()
            raise

    def _receive(self, index: int):
        connection = self._connections[index]
        if not connection.poll(30.0):
            process = self._processes[index]
            raise RuntimeError(f"environment {index} timed out (exitcode={process.exitcode})")
        try:
            status, payload = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"environment {index} exited unexpectedly") from error
        if status != "ok":
            raise RuntimeError(f"environment {index} failed:\n{payload}")
        return payload

    def step(self, forces) -> list[VectorStep]:
        if self._closed:
            raise RuntimeError("vector environment is closed")
        if len(forces) != self.num_envs:
            raise ValueError("one force is required per environment")
        if self._local is not None:
            results = [_advance(self._local, float(forces[0]))]
        else:
            # 모든 환경에 먼저 명령을 보내야 물리 계산이 동시에 진행됩니다.
            for connection, force in zip(self._connections, forces):
                connection.send(("step", float(force)))
            results = [self._receive(index) for index in range(self.num_envs)]
        self.states = [step.next_state for step in results]
        return results

    @property
    def worker_pids(self) -> list[int]:
        return [process.pid for process in self._processes]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._local is not None:
            self._local.close()
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            process.close()
        for connection in self._connections:
            connection.close()

    def __enter__(self) -> "VectorCartPoleEnv":
        return self

    def __exit__(self, *args) -> None:
        self.close()
