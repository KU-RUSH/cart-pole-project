"""저장된 강화학습 모델의 에피소드 한 번을 실행합니다."""

import argparse
from pathlib import Path
import time

import settings
from src.environment import CartPoleEnv
from src.policy import load_run_policy
from src.reward import calculate_reward


def run_simulation(run=None, *, render=True, last=False, seed=None):
    saved = load_run_policy(run, last=last)
    seed = settings.SEED if seed is None else seed
    total_reward = 0.0
    print(f"model={saved.model_path}")
    with CartPoleEnv(saved.config, seed=seed) as env:
        state = env.reset()
        if render and not env.render():
            raise KeyboardInterrupt
        started = time.perf_counter()
        while True:
            _, force = saved.network.choose_action(state)
            result = env.step(force)
            total_reward += calculate_reward(state, result.force, result.state,
                                            result.terminated, saved.config,
                                            getattr(saved.network, "reward_settings", None))
            state = result.state
            if render:
                if not env.render():
                    raise KeyboardInterrupt
                time.sleep(max(0.0, started + state.sim_time - time.perf_counter()))
            if result.done:
                break
    outcome = "완주" if result.truncated and not result.terminated else "실패"
    print(f"{outcome} | time={state.sim_time:.2f}s return={total_reward:.2f} "
          f"x={state.x * 100:+.2f}cm")
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="확인할 results/runs/실행폴더")
    parser.add_argument("--last", action="store_true", help="최고 평가 모델 대신 마지막 모델 실행")
    parser.add_argument("--no-viewer", action="store_true", help="화면 없이 실행")
    args = parser.parse_args(argv)
    try:
        run_simulation(args.run, render=not args.no_viewer, last=args.last)
    except KeyboardInterrupt:
        print("실행을 중단했습니다.")
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"오류: {error}") from error


if __name__ == "__main__":
    main()
