"""settings.py의 설정으로 학습하고 results/에 실행별 결과를 저장합니다."""

import argparse
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import settings
from src.run_results import create_run


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    algorithm = settings.ALGORITHM.lower()
    if algorithm not in ("dqn", "sac", "ppo"):
        raise ValueError("settings.ALGORITHM은 dqn, sac, ppo 중 하나여야 합니다.")
    files = create_run(algorithm)
    common = dict(
        num_envs=settings.NUM_ENVS, seed=settings.SEED,
        duration=settings.DURATION_SECONDS, goal=settings.REWARD_PROFILE,
        smoothing_alpha=settings.SMOOTHING_ALPHA,
        max_force_change=settings.MAX_FORCE_CHANGE,
        eval_episodes=settings.EVAL_EPISODES,
        gif_interval=settings.GIF_INTERVAL, gif_fps=settings.GIF_FPS,
        gif_duration=settings.GIF_DURATION_SECONDS,
        report_interval=settings.REPORT_INTERVAL_SECONDS,
        show_report=settings.OPEN_REPORT, watch=settings.WATCH,
        watch_interval=settings.WATCH_INTERVAL_SECONDS,
    )
    if algorithm == "dqn":
        from src.training.dqn import train
        train(files, episodes=settings.EPISODES, **settings.DQN, **common)
    else:
        from src.training.continuous import train
        train(files, algorithm=algorithm, total_steps=settings.TOTAL_STEPS,
              **settings.CONTINUOUS, **common)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"오류: {error}") from error
