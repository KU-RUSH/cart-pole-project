"""결과 페이지를 열거나, 저장된 모델을 별도 초기 상태들에서 평가합니다."""

import argparse
from dataclasses import asdict
from pathlib import Path

import settings
from src.evaluation import evaluate_network
from src.policy import load_run_policy, resolve_run
from src.run_results import open_report, save_json, write_report


def evaluate_run(run=None, *, last=False):
    saved = load_run_policy(run, last=last)
    stats = evaluate_network(saved.network, saved.config,
                             episodes=settings.EVALUATION_EPISODES,
                             seed=settings.EVALUATION_SEED)
    payload = {
        "model": saved.model_path.name,
        "episodes": settings.EVALUATION_EPISODES,
        "seed": settings.EVALUATION_SEED,
        "duration_seconds": saved.config.duration_seconds,
        **asdict(stats),
    }
    save_json(saved.files.logs / "holdout.json", payload)
    write_report(saved.files)
    print(f"model={saved.model_path}")
    print(f"별도 평가 {payload['episodes']}회 | seed={payload['seed']} | "
          f"완주={stats.success_rate:.1%} 정지={stats.settled_rate:.1%} "
          f"평균 리워드={stats.mean_reward:.2f}")
    print("정지: 마지막 최대 2초 동안 위치 ±5cm, 속도 ±5cm/s, 각도 ±1도, 각속도 ±0.1rad/s")
    print(f"결과={saved.files.logs / 'holdout.json'}")
    return saved.files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["show", "evaluate"], default="show")
    parser.add_argument("--run", type=Path, help="확인할 results/runs/실행폴더")
    parser.add_argument("--last", action="store_true", help="마지막 모델 평가")
    args = parser.parse_args(argv)
    if args.last and args.command != "evaluate":
        parser.error("--last는 evaluate에서 사용합니다.")
    try:
        files = evaluate_run(args.run, last=args.last) if args.command == "evaluate" else resolve_run(args.run)
        write_report(files)
        open_report(files)
        print(f"결과 페이지={files.report}")
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"오류: {error}") from error


if __name__ == "__main__":
    main()
