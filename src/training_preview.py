"""학습과 분리된 프로세스에서 최신 정책과 보상 추이를 보여줍니다."""

from dataclasses import asdict
import io
import math
import multiprocessing as mp
from pathlib import Path
import pickle
from queue import Empty, Full
import time

import mujoco
import numpy as np

from src.config import EnvConfig
from src.environment import CartPoleEnv
from src.reward import calculate_reward
from src.reward_plot import _read_series, trailing_mean


def encode_policy(model) -> tuple[str, bytes]:
    # Queue의 feeder thread가 나중에 가변 가중치를 읽지 않도록 여기서 복사합니다.
    if hasattr(model, "params"):
        return "DQN", pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    buffer = io.BytesIO()
    model.save(buffer)
    return model.cartpole_settings["algorithm"], buffer.getvalue()


def decode_policy(kind: str, payload: bytes):
    if kind == "DQN":
        return pickle.loads(payload)  # 이 프로그램이 생성한 내부 메시지만 받습니다.
    import torch
    from stable_baselines3 import PPO, SAC
    from src.controllers.continuous import ContinuousPolicy
    torch.set_num_threads(1)
    constructor = {"PPO": PPO, "SAC": SAC}[kind]
    return ContinuousPolicy(constructor.load(io.BytesIO(payload), device="cpu"))


class RewardFigure:
    def __init__(self, training_log, evaluation_log):
        self.training_log, self.evaluation_log = Path(training_log), Path(evaluation_log)
        self.figure = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.figure)
        self.figure.title = "Episode reward"
        self.figure.xlabel = "Completed episodes"
        self.figure.flg_legend = True
        self.figure.linename[0] = "Train avg20"
        self.figure.linename[1] = "Evaluation"
        self.figure.linergb[0] = [.9, .45, .1]
        self.figure.linergb[1] = [.2, .55, 1.]
        self.figure.linewidth = 2.
        self.completed = 0

    def refresh(self):
        try:
            episodes, rewards = _read_series(self.training_log, "return")
            evaluated, eval_rewards = _read_series(self.evaluation_log, "mean_return")
        except (OSError, ValueError):
            return  # CSV 기록 도중이거나 시작 직후이면 이전 표시를 유지합니다.
        self.completed = int(episodes[-1]) if len(episodes) else 0
        series = [(episodes, trailing_mean(rewards)), (evaluated, eval_rewards)]
        capacity = self.figure.linedata.shape[1] // 2
        for line, (xs, ys) in enumerate(series):
            count = min(len(xs), capacity)
            self.figure.linepnt[line] = count
            if count:
                selected = np.linspace(0, len(xs) - 1, count, dtype=int)
                self.figure.linedata[line, :2*count:2] = xs[selected]
                self.figure.linedata[line, 1:2*count:2] = ys[selected]
        ys = np.concatenate([values for _, values in series])
        maximum_episode = max(self.completed, int(evaluated[-1]) if len(evaluated) else 0, 1)
        self.figure.range[0] = [0, maximum_episode]
        if len(ys):
            low, high = float(ys.min()), float(ys.max())
            margin = max(1., .1 * (high - low))
            self.figure.range[1] = [low - margin, high + margin]


def newest_message(messages):
    newest = None
    while True:
        try:
            newest = messages.get_nowait()
        except Empty:
            return newest


def run_preview(messages, stop, config_dict, training_log, evaluation_log, seed,
                show_window=True, ready=None):
    """독립된 에피소드를 실제 속도로 실행하며 학습 RNG/replay를 사용하지 않습니다."""
    try:
        message = None
        while message is None and not stop.is_set():
            try:
                message = messages.get(timeout=.2)
            except Empty:
                pass
        if message is None:
            return
        policy = decode_policy(message["algorithm"], message["policy"])
        config = EnvConfig(**config_dict)
        rewards = RewardFigure(training_log, evaluation_log)
        next_graph = 0.
        with CartPoleEnv(config) as env:
            while not stop.is_set():
                latest = newest_message(messages)
                if latest is not None:
                    message = latest
                    policy = decode_policy(message["algorithm"], message["policy"])
                # 같은 초기 상태에서 모델의 변화를 비교합니다.
                state = env.reset(seed=seed)
                total = 0.
                started = time.perf_counter()
                while not stop.is_set():
                    now = time.perf_counter()
                    if now >= next_graph:
                        rewards.refresh()
                        next_graph = now + 1.
                    if show_window:
                        if not env.render():
                            return
                        viewer = env._viewer
                        viewer.set_texts((None, None,
                            "Live training (logged episode)\nPreview snapshot episode\nSnapshot env steps\nPreview mode\nPreview time / return\nClose this window",
                            f"{rewards.completed}\n{message['episode']}\n{message['steps']}\n{message['algorithm']}: no exploration\n{state.sim_time:.1f}s / {total:.1f}\nTraining continues"))
                        viewport = viewer.viewport
                        width, height = max(1, int(viewport.width*.42)), max(1, int(viewport.height*.28))
                        rectangle = mujoco.MjrRect(max(0, viewport.width-width-12), 12, width, height)
                        viewer.set_figures((rectangle, rewards.figure))
                    if ready is not None:
                        ready.set()
                    _, force = policy.choose_action(state)
                    result = env.step(force)
                    total += calculate_reward(state, result.force, result.state, result.terminated,
                                              config, getattr(policy, "reward_settings", None))
                    state = result.state
                    # 중단 요청에 바로 반응하며 학습 프로세스를 기다리게 하지 않습니다.
                    stop.wait(max(0., started + env.step_count*env.control_period - time.perf_counter()))
                    if result.done:
                        stop.wait(.3)
                        break
    except Exception as error:
        print(f"학습 미리보기 종료: {type(error).__name__}: {error} (학습은 계속합니다)", flush=True)
    finally:
        stop.set()


class TrainingPreview:
    def __init__(self, config, training_log, evaluation_log, seed, interval=5., *, show_window=True):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("watch_interval must be finite and > 0")
        context = mp.get_context("spawn")
        self.messages, self.stop = context.Queue(maxsize=2), context.Event()
        self.ready = context.Event()
        self.process = context.Process(target=run_preview,
            args=(self.messages, self.stop, asdict(config), str(training_log), str(evaluation_log), seed, show_window, self.ready),
            daemon=True)
        self.interval, self.next_update = interval, 0.
        self.closed = False

    def start(self):
        try:
            self.process.start()
            print(f"학습 미리보기 PID={self.process.pid}: 최신 정책 + 실시간 보상 그래프", flush=True)
        except (OSError, RuntimeError) as error:
            print(f"학습 미리보기 시작 실패: {error} (학습은 계속합니다)", flush=True)
            self.close()

    @property
    def due(self):
        return (not self.closed and not self.stop.is_set() and self.process.is_alive()
                and time.perf_counter() >= self.next_update)

    def publish(self, model, episode, steps):
        if self.closed or self.stop.is_set() or not self.process.is_alive():
            return
        try:
            kind, payload = encode_policy(model)
        except Exception as error:
            print(f"학습 미리보기 전송 실패: {error} (학습은 계속합니다)", flush=True)
            self.close()
            return
        message = {"algorithm": kind, "policy": payload, "episode": episode, "steps": steps}
        # 화면이 늦어도 학습을 막지 않고 이전 스냅샷을 버립니다.
        newest_message(self.messages)
        try:
            self.messages.put_nowait(message)
        except Full:
            pass
        self.next_update = time.perf_counter() + self.interval

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if self.process.pid is not None:
            self.process.join(timeout=2.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.)
        self.messages.cancel_join_thread()
        self.messages.close()
