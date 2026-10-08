"""실행 폴더의 저장된 설정과 강화학습 정책을 함께 불러옵니다."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.config import EnvConfig
from src.controllers.dqn import QNetwork
from src.evaluation import ForcePolicy
from src.run_results import RunFiles, latest_run, load_run


@dataclass(frozen=True)
class SavedPolicy:
    files: RunFiles
    network: ForcePolicy
    config: EnvConfig
    model_path: Path


def resolve_run(run: str | Path | None = None, algorithm: str | None = None) -> RunFiles:
    if run is not None:
        return load_run(run)
    if algorithm is None:
        import settings
        algorithm = settings.ALGORITHM
    try:
        return latest_run(algorithm)
    except FileNotFoundError as error:
        raise FileNotFoundError("학습 결과가 없습니다. 먼저 python train.py를 실행하세요.") from error


def load_run_policy(run: str | Path | RunFiles | None = None, *, last: bool = False,
                    algorithm: str | None = None) -> SavedPolicy:
    files = run if isinstance(run, RunFiles) else resolve_run(run, algorithm)
    model_path = files.last_model if last else files.best_model
    if not model_path.is_file():
        raise FileNotFoundError(f"저장된 모델이 없습니다: {model_path}. 학습이 시작될 때까지 기다려 주세요.")
    if files.algorithm == "dqn":
        metadata = json.loads(files.settings.read_text(encoding="utf-8"))
        network = QNetwork.load(model_path)
        config = EnvConfig(**metadata["environment"])
        network.reward_settings = metadata.get("reward")
    else:
        from src.controllers.continuous import load_policy
        network = load_policy(model_path)
        config = network.config
    if not np.isclose(network.max_force, config.max_force):
        raise ValueError("저장된 모델과 환경의 최대 힘이 다릅니다.")
    if network.reward_profile != config.reward_profile:
        raise ValueError("저장된 모델과 환경의 리워드 설정이 다릅니다.")
    return SavedPolicy(files, network, config, model_path)
