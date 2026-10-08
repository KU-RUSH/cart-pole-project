"""PPO/SAC가 공유하는 관측·힘 변환·체크포인트 로딩."""

import json
from pathlib import Path
import zipfile

import numpy as np

from src.config import EnvConfig
from src.continuous_environment import ContinuousControl, OBSERVATION_SCALE, observe_state
from src.state import State


class ContinuousPolicy:
    def __init__(self, model):
        if not hasattr(model, "cartpole_settings"):
            raise ValueError("checkpoint is missing cartpole_settings")
        self.model = model
        self.config = EnvConfig(**model.cartpole_settings["environment"])
        self.control = ContinuousControl(**model.cartpole_settings["control"])
        self.max_force = self.config.max_force
        self.reward_profile = self.config.reward_profile
        self.reward_settings = model.cartpole_settings.get("reward")
        self.observation_scale = np.asarray(model.cartpole_settings.get("observation_scale", OBSERVATION_SCALE), dtype=np.float32)
        if self.observation_scale.shape != (5,) or not np.all(np.isfinite(self.observation_scale) & (self.observation_scale > 0)):
            raise ValueError("invalid observation_scale")

    def choose_action(self, state: State) -> tuple[float, float]:
        observation = observe_state(state, self.max_force, self.observation_scale)
        action, _ = self.model.predict(observation, deterministic=True)
        value = float(np.asarray(action).reshape(-1)[0])
        _, force = self.control.apply(state, value, self.max_force)
        return value, force

    def target_force_for_action(self, action: float) -> float:
        return float(np.clip(action, -1, 1)) * self.max_force


def load_policy(path: str | Path) -> ContinuousPolicy:
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    algorithm = data.get("cartpole_settings", {}).get("algorithm", "SAC")
    if algorithm not in ("PPO", "SAC"):
        raise ValueError(f"unsupported checkpoint algorithm: {algorithm}")
    import torch
    from stable_baselines3 import PPO, SAC
    torch.set_num_threads(1)
    constructor = {"PPO": PPO, "SAC": SAC}[algorithm]
    return ContinuousPolicy(constructor.load(path, device="cpu"))
