"""학습/평가 CSV를 읽어 보상 추이를 PNG 또는 SVG로 저장합니다."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


def _read_series(path: Path, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    episodes, values = [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"episode", value_column}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: episode/{value_column} 열이 필요합니다.")
        for line, row in enumerate(reader, start=2):
            if row.get("episode") in (None, "") or row.get(value_column) in (None, ""):
                continue  # 학습 중 읽을 때 아직 완성되지 않은 마지막 행은 제외합니다.
            try:
                episode, value = int(row["episode"]), float(row[value_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path.name}:{line}: 잘못된 보상 데이터입니다.") from error
            if episode < 0 or not np.isfinite(value):
                raise ValueError(f"{path.name}:{line}: 에피소드/보상 값이 유효하지 않습니다.")
            episodes.append(episode)
            values.append(value)
    return np.asarray(episodes), np.asarray(values, dtype=np.float64)


def trailing_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    """미래 값을 사용하지 않는 최근 window회 평균. 초반에는 존재하는 값만 평균."""
    if window < 1:
        raise ValueError("window는 1 이상이어야 합니다.")
    values = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (prefix[ends] - prefix[starts]) / (ends - starts)


def plot_reward_logs(
    training_log: str | Path, evaluation_log: str | Path | None = None,
    output: str | Path | None = None, *, window: int = 20,
) -> Path:
    """CSV/가중치를 수정하지 않고 그래프만 저장합니다. GUI 없이도 실행됩니다."""
    if window < 1:
        raise ValueError("window는 1 이상이어야 합니다.")
    training_log = Path(training_log)
    prefix = training_log.stem.removesuffix("_training")
    if evaluation_log is None:
        candidate = training_log.with_name(prefix + "_evaluation.csv")
        evaluation_log = candidate if candidate.exists() else None
    output = Path(output) if output is not None else training_log.with_name(prefix + "_rewards.png")
    if output.suffix.lower() not in (".png", ".svg"):
        raise ValueError("그래프 출력 경로는 .png 또는 .svg여야 합니다.")
    protected = [training_log.resolve()]
    if evaluation_log is not None:
        protected.append(Path(evaluation_log).resolve())
    if output.resolve() in protected:
        raise ValueError("출력 경로는 CSV 로그와 달라야 합니다.")
    episodes, rewards = _read_series(training_log, "return")
    eval_episodes, eval_rewards = np.array([]), np.array([])
    if evaluation_log is not None:
        eval_episodes, eval_rewards = _read_series(Path(evaluation_log), "mean_return")
    if not len(rewards) and not len(eval_rewards):
        raise ValueError("아직 그래프로 그릴 보상 기록이 없습니다.")

    # import 시에는 Matplotlib을 로드하지 않아 병렬 환경 프로세스 시작에 영향이 없습니다.
    cache = Path(__file__).resolve().parents[1] / "results" / ".matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise RuntimeError("그래프를 그리려면 python -m pip install -r requirements.txt를 실행하세요.") from error

    figure = Figure(figsize=(11, 6), dpi=160, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_axes((0.09, 0.16, 0.88, 0.64))
    axis.set_facecolor("#fafbfd")
    if len(rewards):
        axis.plot(episodes, rewards, color="#9aa5b1", alpha=0.55, linewidth=0.85,
                  label="Training: each episode")
        axis.plot(episodes, trailing_mean(rewards, window), color="#e47a20", linewidth=2.3,
                  label=f"Training: last {window} mean")
    if len(eval_rewards):
        axis.plot(eval_episodes, eval_rewards, color="#2563a6", linewidth=2.0,
                  marker="o", markersize=4.5, label="Evaluation: no exploration")
    axis.set_xlabel("Completed training episodes", fontsize=11, labelpad=10)
    axis.set_ylabel("Total reward (episode return)", fontsize=11, labelpad=10)
    axis.grid(True, color="#dce2e8", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#aab4be")
    axis.tick_params(colors="#43505c", labelsize=10)
    max_episode = max(int(episodes.max()) if len(episodes) else 0,
                      int(eval_episodes.max()) if len(eval_episodes) else 0)
    axis.set_xlim(0, max(1, max_episode))
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3,
                frameon=False, fontsize=9, borderaxespad=0)
    figure.text(0.09, 0.955, "Reward per episode", fontsize=19, weight="bold", color="#172b3a")
    figure.text(0.09, 0.90, f"{prefix}  |  {len(rewards)} completed episodes",
                fontsize=10, color="#526272")
    figure.text(0.09, 0.045, "Training includes exploration. Evaluation uses a separate environment without learning.",
                fontsize=9, color="#526272")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        figure.savefig(temporary, format=output.suffix.lstrip(".").lower(), dpi=160)
        temporary.replace(output)
    finally:
        figure.clear()
    return output
