"""실제 정책을 별도 MuJoCo 환경에서 실행해 GIF와 같은 실행의 CSV를 저장합니다."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import math

import numpy as np

from src.config import EnvConfig
from src.environment import CartPoleEnv
from src.evaluation import ForcePolicy
from src.reward import calculate_reward


def render_episode(
    policy: ForcePolicy, config: EnvConfig, output: str | Path, *, seed: int = 20042,
    duration: float = 10., fps: int = 10, label: str | None = None,
    trace_path: str | Path | None = None,
) -> Path:
    """탐험 없는 실제 실행. 실패하면 즉시 끝내며 화면을 정지시켜 성공처럼 만들지 않습니다."""
    output = Path(output)
    if output.suffix.lower() != ".gif":
        raise ValueError("GIF output must end in .gif")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not isinstance(fps, int) or fps < 1 or fps > config.control_hz:
        raise ValueError("fps must be an integer between 1 and control_hz")
    interval = config.control_hz / fps
    if not math.isclose(interval, round(interval)):
        raise ValueError("fps must divide control_hz")
    if not np.isclose(policy.max_force, config.max_force) or policy.reward_profile != config.reward_profile:
        raise ValueError("policy and environment force/reward settings must match")

    import mujoco
    from PIL import Image, ImageDraw

    config = replace(config, duration_seconds=duration)
    frames, trace = [], []
    with CartPoleEnv(config, seed=seed) as env, mujoco.Renderer(env.model, height=360, width=640) as renderer:
        state = env.reset()

        def capture(ending: str = ""):
            renderer.update_scene(env.data, camera="side")
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 47), fill=(245, 247, 250))
            draw.text((10, 7), f"{label or 'Policy preview'} | seed={seed} | t={state.sim_time:.2f}s {ending}",
                      fill=(20, 30, 40))
            draw.text((10, 27), f"x={state.x*100:+.2f}cm  v={state.x_dot*100:+.2f}cm/s  "
                      f"angle={np.rad2deg(state.theta):+.2f}deg  force={state.previous_force:+.3f}N",
                      fill=(20, 30, 40))
            frames.append(frame)

        capture()
        while True:
            previous = state
            action, force = policy.choose_action(previous)
            result = env.step(force)
            state = result.state
            reward = calculate_reward(previous, result.force, state, result.terminated, config,
                                      getattr(policy, "reward_settings", None))
            trace.append([1, env.step_count, state.sim_time, state.x, state.theta, state.x_dot,
                          state.theta_dot, policy.target_force_for_action(action), result.force,
                          reward, result.terminated, result.truncated])
            if env.step_count % round(interval) == 0 or result.done:
                capture("| FAILED" if result.terminated else "| time limit" if result.truncated else "")
            if result.done:
                break

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        frames[0].save(temporary, format="GIF", save_all=True, append_images=frames[1:],
                       duration=round(1000 / fps), loop=0)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_path.with_name(trace_path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "step", "sim_time", "x", "theta", "x_dot", "theta_dot",
                             "target_force", "applied_force", "reward", "terminated", "truncated"])
            writer.writerows(trace)
        temporary.replace(trace_path)
    return output
