"""학습 한 번마다 폴더 하나: 모델, 로그, 그림, GIF와 로컬 결과 페이지."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
import time
import webbrowser

import numpy as np

from src.config import EnvConfig
from src.evaluation import ForcePolicy
from src.rendering import render_episode
from src.reward_plot import plot_reward_logs


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
LOCAL_TIMEZONE = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class RunFiles:
    root: Path
    algorithm: str

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "algorithm", self.algorithm.lower())
        if self.algorithm not in ("dqn", "sac", "ppo"):
            raise ValueError("algorithm must be dqn, sac or ppo")

    @property
    def models(self): return self.root / "models"
    @property
    def logs(self): return self.root / "logs"
    @property
    def plots(self): return self.root / "plots"
    @property
    def gifs(self): return self.root / "gifs"
    @property
    def best_model(self): return self.models / ("best.npz" if self.algorithm == "dqn" else "best.zip")
    @property
    def last_model(self): return self.models / ("last.npz" if self.algorithm == "dqn" else "last.zip")
    @property
    def training_log(self): return self.logs / "training.csv"
    @property
    def evaluation_log(self): return self.logs / "evaluation.csv"
    @property
    def settings(self): return self.root / "settings.json"
    @property
    def best_info(self): return self.root / "best.json"
    @property
    def report(self): return self.root / "index.html"
    @property
    def state(self): return self.root / "state.json"


def _json_value(value):
    if is_dataclass(value): return _json_value(asdict(value))
    if isinstance(value, dict): return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray): return _json_value(value.tolist())
    if isinstance(value, np.generic): return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def _save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4: raise
            time.sleep(.05 * (attempt + 1))


def save_json(path: str | Path, value):
    _save_text(Path(path), json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def create_run(algorithm: str, name: str | None = None, results_root: str | Path | None = None) -> RunFiles:
    results_root = Path(results_root or RESULTS_ROOT).resolve()
    algorithm = algorithm.lower()
    RunFiles(results_root, algorithm)  # 이름을 검증한 다음 디렉터리를 만듭니다.
    suffix = re.sub(r"[^\w-]+", "_", name or "", flags=re.UNICODE).strip("_")[:60]
    stamp = datetime.now(LOCAL_TIMEZONE).strftime("%Y%m%d_%H%M%S_%f")
    folder = stamp + "_" + algorithm + ("_" + suffix if suffix else "")
    files = RunFiles(results_root / "runs" / folder, algorithm)
    files.root.mkdir(parents=True, exist_ok=False)
    for directory in (files.models, files.logs, files.plots, files.gifs):
        directory.mkdir()
    save_json(files.state, {"algorithm": algorithm, "created_at": datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds"),
                            "episode": 0, "steps": 0, "status": "starting"})
    latest = _read_json(results_root / "latest.json")
    relative = files.root.relative_to(results_root).as_posix()
    latest["latest"] = relative
    latest.setdefault("runs", {})[algorithm] = relative
    save_json(results_root / "latest.json", latest)
    write_report(files)
    return files


def load_run(path: str | Path) -> RunFiles:
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {path}")
    state = _read_json(path / "state.json")
    algorithm = state.get("algorithm") or _read_json(path / "settings.json").get("algorithm")
    if algorithm is None:
        raise ValueError(f"run algorithm is missing: {path}")
    return RunFiles(path, algorithm)


def latest_run(algorithm: str | None = None, results_root: str | Path | None = None) -> RunFiles:
    results_root = Path(results_root or RESULTS_ROOT).resolve()
    latest = _read_json(results_root / "latest.json")
    if algorithm is not None and algorithm.lower() not in ("dqn", "sac", "ppo"):
        raise ValueError("algorithm must be dqn, sac or ppo")
    relative = latest.get("runs", {}).get(algorithm.lower()) if algorithm else latest.get("latest")
    if not relative:
        raise FileNotFoundError("No matching run. Start training first.")
    path = (results_root / relative).resolve()
    if not path.is_relative_to(results_root):
        raise ValueError("latest run must be inside results_root")
    return load_run(path)


def _rows(path: Path):
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _figure(figsize=(10, 6)):
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_ROOT / ".matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=figsize, dpi=120, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _save_figure(figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        figure.savefig(temporary, format="png")
        temporary.replace(path)
    finally:
        figure.clear()
        temporary.unlink(missing_ok=True)


def _waiting_plot(path: Path, title: str, message: str):
    figure = _figure()
    axis = figure.subplots()
    axis.set_title(title)
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    _save_figure(figure, path)


def _number(row, key):
    try: return float(row.get(key, "nan"))
    except (ValueError, TypeError): return float("nan")


def plot_performance(evaluation_log: str | Path, output: str | Path) -> Path:
    """보상과 별개인 생존율, 정지율, 실제 오차/힘. 생존한 실행의 tail RMS만 표시합니다."""
    rows = _rows(Path(evaluation_log))
    output = Path(output)
    if not rows:
        _waiting_plot(output, "Evaluation performance", "Waiting for evaluation records")
        return output
    episodes = [_number(row, "episode") for row in rows]
    figure = _figure((11, 8))
    axes = figure.subplots(2, 3).flat
    axis = next(axes)
    axis.plot(episodes, [_number(r, "success_rate") * 100 for r in rows], label="Survived")
    axis.plot(episodes, [_number(r, "settled_rate") * 100 for r in rows], label="Settled")
    axis.set_ylabel("Evaluation episodes (%)")
    axis.set_ylim(-3, 103)
    axis.legend(fontsize=8)
    for axis, key, scale, label in zip(axes,
            ["tail_position_rms", "tail_velocity_rms", "tail_angle_rms_degrees", "mean_abs_force", "mean_force_change"],
            [100, 100, 1, 1, 1],
            ["Tail position RMS (cm)", "Tail speed RMS (cm/s)", "Tail angle RMS (deg)",
             "Mean absolute force (N)", "Mean force change (N/step)"]):
        values = [_number(r, key) * scale for r in rows]
        axis.plot(episodes, values, color="#2474b5", marker=".")
        axis.set_ylabel(label)
    for axis in figure.axes:
        axis.set_xlabel("Completed training episodes")
        axis.grid(alpha=.2)
    figure.suptitle("Separate evaluation, no exploration | tail RMS: survivors only")
    _save_figure(figure, output)
    return output


def plot_motion(trace_path: str | Path, output: str | Path, episode: int = 1) -> Path:
    rows = [row for row in _rows(Path(trace_path)) if int(row["episode"]) == episode]
    if not rows: raise ValueError(f"episode {episode} has no trace records")
    figure = _figure((10, 8))
    axes = figure.subplots(4, 1, sharex=True)
    times = [_number(row, "sim_time") for row in rows]
    for index, (axis, key, scale, label) in enumerate(zip(axes,
            ["x", "x_dot", "theta", "applied_force"], [100, 100, 180 / np.pi, 1],
            ["Cart position (cm)", "Cart speed (cm/s)", "Pole angle (deg)", "Applied force (N)"])):
        axis.plot(times, [_number(row, key) * scale for row in rows], color="#2474b5")
        axis.axhline(0, linewidth=.7, color="#333", alpha=.5)
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
        if index < 3:
            tolerance = [5, 5, 1][index]
            axis.axhspan(-tolerance, tolerance, color="#32a852", alpha=.1)
    axes[-1].set_xlabel("Simulation time (s)")
    ending = ("failed" if rows[-1].get("terminated", "").lower() == "true" else
              "time limit" if rows[-1].get("truncated", "").lower() == "true" else "unfinished trace")
    figure.suptitle(f"Deterministic preview | episode {episode} | {ending}")
    output = Path(output)
    _save_figure(figure, output)
    return output


def update_artifacts(files: RunFiles, policy: ForcePolicy, config: EnvConfig, episode: int, steps: int,
                     stats=None, *, render_gif: bool = False, status: str = "running",
                     preview_seed: int = 20042, gif_duration: float = 10., gif_fps: int = 10,
                     policy_label: str = "current"):
    """로그 flush 후 호출합니다. 렌더링 불가 시 이유를 결과 페이지에 남깁니다."""
    if policy_label not in ("current", "best"):
        raise ValueError("policy_label must be current or best")
    state = _read_json(files.state)
    best_info = _read_json(files.best_info) if policy_label == "best" else {}
    best_stats = best_info.get("stats", best_info)
    selected_episode = best_info.get("episode", episode)
    state.update(algorithm=files.algorithm, episode=int(episode), steps=int(steps), status=status,
                 updated_at=datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds"), environment=asdict(config))
    if stats is not None:
        # 최고 모델에는 마지막 정책 점수를 붙이지 않습니다. 저장 당시 평가를 사용합니다.
        state["evaluation"] = _json_value(best_stats if "mean_reward" in best_stats else stats)
        state["evaluation_policy_label"] = policy_label
        if policy_label == "best": state["evaluation_episode"] = selected_episode
    for log, header in [(files.training_log, "episode,return\n"),
                        (files.evaluation_log, "episode,mean_return\n")]:
        if not log.exists(): _save_text(log, header)
    graph_errors = {}

    def try_graph(name, draw):
        try:
            draw()
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            graph_errors[name] = f"{type(error).__name__}: {error}"
            print(f"Graph export failed ({name}): {graph_errors[name]}", flush=True)

    def rewards():
        if _rows(files.training_log) or _rows(files.evaluation_log):
            plot_reward_logs(files.training_log, files.evaluation_log, files.plots / "rewards.png")
        else:
            _waiting_plot(files.plots / "rewards.png", "Reward per episode", "Waiting for completed episodes/evaluation")

    try_graph("rewards", rewards)
    try_graph("performance", lambda: plot_performance(files.evaluation_log, files.plots / "performance.png"))
    if render_gif:
        prefix = "best_" if policy_label == "best" else ""
        gif = files.gifs / f"{prefix}episode_{episode:06d}.gif"
        trace = files.logs / "preview.csv"
        try:
            policy_note = (f"best selected ep={selected_episode} | export ep={episode}"
                           if policy_label == "best" else f"current ep={episode}")
            render_episode(policy, config, gif, seed=preview_seed, duration=gif_duration, fps=gif_fps,
                           label=f"{files.algorithm.upper()} {policy_note}", trace_path=trace)
            rows = _rows(trace)
            state["preview"] = {"episode": int(episode), "seed": preview_seed, "policy_label": policy_label,
                                "model_episode": selected_episode if policy_label == "best" else int(episode),
                                "gif": gif.relative_to(files.root).as_posix(),
                                "duration": _number(rows[-1], "sim_time"),
                                "failed": rows[-1]["terminated"].lower() == "true"}
            if policy_label == "current": state["current_preview"] = state["preview"]
            state.pop("preview_error", None)
            try_graph("motion", lambda: plot_motion(trace, files.plots / "motion.png"))
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            state["preview_error"] = f"{type(error).__name__}: {error}"
            print(f"GIF export failed: {state['preview_error']}", flush=True)
    if not (files.plots / "motion.png").exists():
        try_graph("motion", lambda: _waiting_plot(files.plots / "motion.png", "Policy motion",
                                                  "Waiting for a rendered policy preview"))
    state["graph_errors"] = graph_errors
    save_json(files.state, state)
    write_report(files)
    return files.report


def _metric_table(stats):
    if not stats: return ""
    labels = [("mean_reward", "평균 보상", 1), ("success_rate", "생존율 (%)", 100),
              ("settled_rate", "정지 기준 만족률 (%)", 100),
              ("tail_position_rms", "마지막 최대 2초 위치 RMS (cm)", 100),
              ("tail_velocity_rms", "마지막 최대 2초 속도 RMS (cm/s)", 100),
              ("tail_angle_rms_degrees", "마지막 최대 2초 각도 RMS (도)", 1)]
    cells = []
    for key, label, scale in labels:
        value = stats.get(key)
        shown = f"{value * scale:.3g}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"
        cells.append(f"<tr><th>{label}</th><td>{shown}</td></tr>")
    return "<table>" + "".join(cells) + "</table>"


STYLE = """body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#24364b;margin:24px auto;max-width:1200px;padding:0 18px}h1{font-size:24px}h2{font-size:18px}a{color:#215fa8}section{background:white;padding:16px;border-radius:10px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}img{width:100%;height:auto}p{line-height:1.6}table{border-collapse:collapse}th,td{padding:7px 16px 7px 0;text-align:left}small{color:#637387}.error{color:#a43c30}li{margin:10px 0}"""


def write_report(files: RunFiles, status: str | None = None) -> Path:
    state = _read_json(files.state)
    if status is not None:
        state["status"] = status
        save_json(files.state, state)
    status = state.get("status", "starting")
    refresh = '<meta http-equiv="refresh" content="15">' if status in ("starting", "running") else ""
    preview = state.get("preview", {})
    gif = preview.get("gif")
    gif_version = (files.root / gif).stat().st_mtime_ns if gif and (files.root / gif).is_file() else 0
    gif_html = (f'<img src="{escape(gif, quote=True)}?v={gif_version}" alt="실제 정책 실행 GIF">'
                if gif and (files.root / gif).is_file() else "<p>첫 GIF를 기다리는 중입니다.</p>")
    preview_note = (f"{'최고 모델' if preview.get('policy_label') == 'best' else '현재 모델'} · "
                    + (f"선택된 학습 시점: {preview.get('model_episode')}회 · "
                       if preview.get('policy_label') == 'best' else "")
                    + f"내보낸 시점: 완료 {preview.get('episode')}회 · seed {preview.get('seed')} · "
                    f"실제 길이 {preview.get('duration', 0):.2f}초 · "
                    f"{'실패로 종료' if preview.get('failed') else '시간 제한 종료'}") if preview else ""
    error = f'<p class="error">GIF 내보내기 오류: {escape(state["preview_error"])}</p>' if state.get("preview_error") else ""
    graph_error = "".join(f'<p class="error">그래프 저장 오류 ({escape(name)}): {escape(message)}</p>'
                          for name, message in state.get("graph_errors", {}).items())
    images = "".join(f'<section><h2>{title}</h2><img src="plots/{name}.png?v={(files.plots / (name + ".png")).stat().st_mtime_ns}" alt="{title}"></section>'
                     for name, title in [("rewards", "보상 추이"), ("performance", "평가 성능"), ("motion", "GIF와 같은 실행의 움직임")]
                     if (files.plots / f"{name}.png").is_file())
    links = " · ".join(f'<a href="{p.relative_to(files.root).as_posix()}">{label}</a>' for p, label in
                        [(files.best_model, "최고 모델"), (files.last_model, "마지막 모델"),
                         (files.training_log, "학습 CSV"), (files.evaluation_log, "평가 CSV"),
                         (files.settings, "설정"), (files.best_info, "최고 평가")]
                        if p.is_file())
    holdout = _read_json(files.logs / "holdout.json")
    holdout_note = " · ".join(f"{label}: {escape(str(holdout[key]))}" for key, label in
                              [("model", "모델"), ("episodes", "실행 수"), ("seed", "seed"),
                               ("duration_seconds", "실행 길이(초)"), ("duration", "실행 길이(초)")]
                              if key in holdout)
    holdout_html = ('<section><h2>별도 모델 평가</h2><small>' + holdout_note + '</small>'
                    + _metric_table(holdout.get("stats", holdout)) + '</section>') if holdout else ""
    evaluation_title = "최고 모델 평가" if state.get("evaluation_policy_label") == "best" else "최근 학습 중 평가"
    duration = state.get("environment", {}).get("duration_seconds")
    duration_note = f"평가 실행 길이: {duration:g}초." if isinstance(duration, (int, float)) else ""
    current = state.get("current_preview", {})
    current_link = (f'<p><a href="{escape(current["gif"], quote=True)}">학습 중 마지막 현재 모델 GIF '
                    f'({current.get("episode", 0)}회)</a></p>') if preview.get("policy_label") == "best" and current else ""
    index_link = '<a href="../../index.html">전체 학습 목록</a> · ' if files.root.parent.name == "runs" else ""
    html = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">{refresh}<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(files.root.name)}</title><style>{STYLE}</style></head><body>
<h1>{files.algorithm.upper()} 학습 결과</h1><p>{escape(files.root.name)}<br>상태: {escape(status)} · 완료 에피소드: {state.get('episode', 0)} · 환경 스텝: {state.get('steps', 0)}</p>
<p>{links}</p><div class="grid"><section><h2>실제 정책 미리보기</h2>{gif_html}<small>{preview_note}</small>{current_link}{error}<p>탐험 없이 별도 환경에서 실행한 정책입니다. GIF 한 편은 전체 평가 성능을 대표하지 않습니다.</p></section>
<section><h2>{evaluation_title}</h2>{_metric_table(state.get('evaluation'))}<p>{duration_note}<br>생존: 설정한 시간까지 실패하지 않음.<br>정지: 마지막 최대 2초 내내 위치 ±5cm, 속도 ±5cm/s, 각도 ±1도, 각속도 ±0.1rad/s.<br>마지막 최대 2초 RMS는 생존한 실행만 계산합니다.</p></section></div>
{graph_error}<div class="grid">{images}</div>{holdout_html}<p>{index_link}학습 중인 페이지는 15초마다 새로고침됩니다.</p></body></html>'''
    _save_text(files.report, html)
    if files.root.parent.name == "runs":
        _write_index(files.root.parent.parent)
    return files.report


def _write_index(results_root: Path):
    entries = []
    runs = results_root / "runs"
    for path in sorted(runs.iterdir(), reverse=True) if runs.exists() else []:
        if not path.is_dir() or not (path / "state.json").is_file(): continue
        state = _read_json(path / "state.json")
        link = path.relative_to(results_root).as_posix() + "/index.html"
        entries.append(f'<li><a href="{escape(link, quote=True)}">{escape(path.name)}</a> · '
                       f'{escape(state.get("status", "unknown"))} · {state.get("episode", 0)} episodes</li>')
    _save_text(results_root / "index.html", f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>학습 결과</title><style>{STYLE}</style></head><body><h1>강화학습 결과</h1><p>학습마다 새 폴더를 만들고 기존 결과를 보존합니다.</p><ul>{"".join(entries)}</ul></body></html>')


def open_report(files: RunFiles):
    webbrowser.open(files.report.as_uri())
