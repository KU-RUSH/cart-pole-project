"""여러 MuJoCo 환경의 경험으로 하나의 Double DQN을 학습합니다."""

from collections import deque
import csv
from dataclasses import asdict
import math
import os
import time

import numpy as np

from src.config import EnvConfig
from src.controllers.dqn import DQNAgent, QNetwork
from src.evaluation import evaluate_network
from src.reward import reward_settings
from src.run_results import RunFiles, save_json, update_artifacts, open_report
from src.training_preview import TrainingPreview
from src.vector_environment import VectorCartPoleEnv


def train(files: RunFiles, *, episodes=1200, num_envs=1, seed=42,
          learning_rate=.0005, gamma=.995, epsilon_decay_steps=50_000,
          epsilon_min=.02, eval_interval=100, eval_episodes=10,
          duration=10., goal="still", smoothing_alpha=.25, max_force_change=2.,
          gif_interval=100, gif_fps=10, gif_duration=10., report_interval=5.,
          show_report=True, watch=False, watch_interval=5.):
    if files.algorithm != "dqn":
        raise ValueError("DQN needs a dqn run folder")
    if min(episodes, num_envs, epsilon_decay_steps, eval_interval, eval_episodes, gif_interval) < 1:
        raise ValueError("episode/environment/interval counts must be >= 1")
    if seed < 0 or not np.isfinite(epsilon_min) or not 0 <= epsilon_min <= 1:
        raise ValueError("seed must be >= 0; epsilon_min must be in [0, 1]")
    if any(not math.isfinite(value) or value <= 0 for value in
           (report_interval, watch_interval, gif_duration)):
        raise ValueError("time intervals must be finite and positive")
    config = EnvConfig(duration_seconds=duration, reward_profile=goal)
    agent = DQNAgent(config.max_force, learning_rate=learning_rate, seed=seed,
                     smoothing_alpha=smoothing_alpha, max_force_change=max_force_change,
                     reward_profile=goal, gamma=gamma)
    reward = reward_settings(goal)
    agent.online.reward_settings = reward
    save_json(files.settings, {
        "algorithm": "dqn", "environment": asdict(config),
        "training": {"episodes": episodes, "num_envs": num_envs, "seed": seed,
            "learning_rate": learning_rate, "gamma": gamma,
            "epsilon_decay_steps": epsilon_decay_steps, "epsilon_min": epsilon_min,
            "eval_interval": eval_interval, "eval_episodes": eval_episodes,
            "evaluation_seed": seed + 10_000, "replay_capacity": len(agent.replay.actions)},
        "control": {"smoothing_alpha": smoothing_alpha, "max_force_change": max_force_change},
        "action_forces": agent.online.action_forces, "state_scale": agent.online.state_scale,
        "reward": reward, "artifacts": {"gif_interval": gif_interval, "gif_fps": gif_fps,
            "gif_duration": gif_duration, "report_interval": report_interval,
            "preview_seed": seed + 20_000, "watch": watch, "watch_interval": watch_interval},
    })
    completed = 0
    next_eval, next_gif = eval_interval, gif_interval
    last_eval, last_gif = -1, -1
    best_key = (-math.inf,) * 4
    recent_rewards, recent_steps = deque(maxlen=20), deque(maxlen=20)
    started = time.perf_counter()
    last_report = started
    stats = None
    status = "failed"
    preview = (TrainingPreview(config, files.training_log, files.evaluation_log,
                               seed + 20_000, watch_interval) if watch else None)

    with files.training_log.open("w", newline="", encoding="utf-8") as training_file, \
            files.evaluation_log.open("w", newline="", encoding="utf-8") as evaluation_file:
        training_writer, evaluation_writer = csv.writer(training_file), csv.writer(evaluation_file)
        training_writer.writerow(["episode", "env_index", "return", "steps", "epsilon", "loss",
                                 "environment_steps", "learning_steps", "elapsed_seconds"])
        evaluation_writer.writerow(["episode", "environment_steps", "mean_return", "mean_steps",
            "success_rate", "elapsed_seconds", "mean_abs_force", "mean_force_change", "max_force_change",
            "tail_position_rms", "tail_velocity_rms", "tail_angle_rms_degrees",
            "tail_angular_velocity_rms", "settled_rate"])

        def export(*, gif=False, final_status="running", policy=None, policy_label="current"):
            nonlocal last_report, last_gif
            training_file.flush()
            evaluation_file.flush()
            update_artifacts(files, policy or agent.online, config, completed, agent.environment_steps,
                stats, render_gif=gif, status=final_status, preview_seed=seed + 20_000,
                gif_duration=min(duration, gif_duration), gif_fps=gif_fps, policy_label=policy_label)
            last_report = time.perf_counter()
            if gif:
                last_gif = completed

        def evaluate():
            nonlocal best_key, stats, last_eval
            stats = evaluate_network(agent.online, config, eval_episodes, seed=seed + 10_000)
            last_eval = completed
            evaluation_writer.writerow([completed, agent.environment_steps, stats.mean_reward,
                stats.mean_steps, stats.success_rate, time.perf_counter() - started,
                stats.mean_abs_force, stats.mean_abs_force_change, stats.max_abs_force_change,
                stats.tail_position_rms, stats.tail_velocity_rms, stats.tail_angle_rms_degrees,
                stats.tail_angular_velocity_rms, stats.settled_rate])
            evaluation_file.flush()
            key = (stats.success_rate, stats.mean_steps, stats.settled_rate, stats.mean_reward)
            if key > best_key:
                best_key = key
                agent.save(files.best_model)
                save_json(files.best_info, {"episode": completed,
                    "environment_steps": agent.environment_steps,
                    "evaluation_seed": seed + 10_000, **asdict(stats)})
            agent.save(files.last_model)
            print(f"평가 | episode={completed} return={stats.mean_reward:.2f} "
                  f"생존={stats.success_rate:.0%} 정지={stats.settled_rate:.0%}", flush=True)

        print(f"DQN | environments={num_envs} seed={seed} episodes={episodes}\n"
              f"결과: {files.root}", flush=True)
        try:
            evaluate()
            export(gif=True)
            if show_report:
                open_report(files)
            if preview:
                preview.start()
                preview.publish(agent.online, completed, agent.environment_steps)
            with VectorCartPoleEnv(config, num_envs, seed) as envs:
                print(f"worker_pids={envs.worker_pids or [os.getpid()]}", flush=True)
                returns = np.zeros(num_envs)
                while completed < episodes:
                    epsilon = max(epsilon_min,
                        1.0 - (1.0 - epsilon_min) * agent.environment_steps / epsilon_decay_steps)
                    states = envs.states
                    actions, forces = agent.choose_actions(states, epsilon)
                    results = envs.step(forces)
                    for index, step in enumerate(results):
                        # 시간 제한은 실패가 아니므로 terminated만 TD 종료 마스크로 씁니다.
                        agent.observe(states[index], int(actions[index]), step.reward,
                                      step.result.state, step.result.terminated)
                        returns[index] += step.reward
                    losses = []
                    while (loss := agent.learn()) is not None:
                        losses.append(loss)
                    mean_loss = float(np.mean(losses)) if losses else 0.0
                    for index, step in enumerate(results):
                        if step.result.done and completed < episodes:
                            completed += 1
                            recent_rewards.append(float(returns[index]))
                            recent_steps.append(step.episode_steps)
                            training_writer.writerow([completed, index, returns[index], step.episode_steps,
                                epsilon, mean_loss, agent.environment_steps, agent.learning_steps,
                                time.perf_counter() - started])
                            returns[index] = 0.
                            if completed == 1 or completed % 10 == 0:
                                training_file.flush()
                                print(f"episode={completed}/{episodes} return20={np.mean(recent_rewards):.2f} "
                                      f"steps20={np.mean(recent_steps):.1f} epsilon={epsilon:.3f}", flush=True)
                    evaluated = completed >= next_eval
                    if evaluated:
                        evaluate()
                        next_eval = (completed // eval_interval + 1) * eval_interval
                    gif_due = completed >= next_gif
                    if gif_due:
                        next_gif = (completed // gif_interval + 1) * gif_interval
                    if gif_due or evaluated or time.perf_counter() - last_report >= report_interval:
                        export(gif=gif_due)
                    if preview and preview.due:
                        training_file.flush()
                        preview.publish(agent.online, completed, agent.environment_steps)
            if completed != last_eval:
                evaluate()
            status = "complete"
        except KeyboardInterrupt:
            status = "interrupted"
            print("학습 중단: 마지막 가중치를 저장합니다.", flush=True)
        finally:
            try:
                agent.save(files.last_model)
                export(gif=last_gif != completed, final_status=status)
                if files.best_model.exists():
                    best = QNetwork.load(files.best_model)
                    best.reward_settings = reward
                    export(gif=True, final_status=status, policy=best, policy_label="best")
            finally:
                if preview:
                    preview.close()
    print(f"결과 페이지: {files.report}\nBest: {files.best_model}\nLast: {files.last_model}", flush=True)
    return agent
