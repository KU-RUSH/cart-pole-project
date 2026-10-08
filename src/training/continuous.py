"""Gymnasium 환경과 Stable-Baselines3 SAC/PPO를 연결하는 학습 루프."""

import csv
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time

import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.config import EnvConfig
from src.continuous_environment import ContinuousCartPoleEnv, ContinuousControl, OBSERVATION_SCALE
from src.controllers.continuous import ContinuousPolicy, load_policy
from src.evaluation import evaluate_network
from src.reward import reward_settings
from src.run_results import RunFiles, save_json, update_artifacts, open_report
from src.training_preview import TrainingPreview


@dataclass(frozen=True)
class EnvFactory:
    config: EnvConfig
    control: ContinuousControl
    reward: dict

    def __call__(self):
        return Monitor(ContinuousCartPoleEnv(self.config, self.control, settings=self.reward))


def save_model(model, path: Path):
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


class ResultCallback(BaseCallback):
    def __init__(self, files, config, *, seed, eval_interval, eval_episodes,
                 gif_interval, gif_duration, gif_fps, report_interval,
                 show_report, watch, watch_interval):
        super().__init__()
        self.files, self.config = files, config
        self.seed, self.eval_interval, self.eval_episodes = seed, eval_interval, eval_episodes
        self.gif_interval, self.gif_duration, self.gif_fps = gif_interval, min(config.duration_seconds, gif_duration), gif_fps
        self.report_interval, self.show_report = report_interval, show_report
        self.started = self.last_report = time.perf_counter()
        self.completed = 0
        self.next_eval, self.next_gif = eval_interval, gif_interval
        self.last_eval, self.last_gif = -1, -1
        self.best_key, self.stats = (-math.inf,) * 4, None
        self.preview = (TrainingPreview(config, files.training_log, files.evaluation_log,
                        seed + 20_000, watch_interval) if watch else None)
        self.training_file = files.training_log.open("w", newline="", encoding="utf-8")
        self.evaluation_file = files.evaluation_log.open("w", newline="", encoding="utf-8")
        self.training_writer, self.evaluation_writer = csv.writer(self.training_file), csv.writer(self.evaluation_file)
        self.training_writer.writerow(["episode", "env_index", "return", "steps", "environment_steps", "elapsed_seconds"])
        self.evaluation_writer.writerow(["episode", "environment_steps", "mean_return", "mean_steps",
            "success_rate", "settled_rate", "tail_position_rms", "tail_velocity_rms",
            "tail_angle_rms_degrees", "tail_angular_velocity_rms", "mean_abs_force",
            "mean_force_change", "max_force_change", "elapsed_seconds"])

    def export(self, *, gif=False, status="running", policy=None, policy_label="current"):
        self.training_file.flush()
        self.evaluation_file.flush()
        update_artifacts(self.files, policy or ContinuousPolicy(self.model), self.config,
            self.completed, self.model.num_timesteps, self.stats, render_gif=gif,
            status=status, preview_seed=self.seed + 20_000, gif_duration=self.gif_duration,
            gif_fps=self.gif_fps, policy_label=policy_label)
        self.last_report = time.perf_counter()
        if gif:
            self.last_gif = self.completed

    def evaluate(self):
        stats = evaluate_network(ContinuousPolicy(self.model), self.config, self.eval_episodes,
                                 seed=self.seed + 10_000)
        self.stats, self.last_eval = stats, self.model.num_timesteps
        self.evaluation_writer.writerow([self.completed, self.model.num_timesteps,
            stats.mean_reward, stats.mean_steps, stats.success_rate, stats.settled_rate,
            stats.tail_position_rms, stats.tail_velocity_rms, stats.tail_angle_rms_degrees,
            stats.tail_angular_velocity_rms, stats.mean_abs_force, stats.mean_abs_force_change,
            stats.max_abs_force_change, time.perf_counter() - self.started])
        self.evaluation_file.flush()
        key = (stats.success_rate, stats.mean_steps, stats.settled_rate, stats.mean_reward)
        if key > self.best_key:
            self.best_key = key
            save_model(self.model, self.files.best_model)
            save_json(self.files.best_info, {"episode": self.completed,
                "environment_steps": self.model.num_timesteps,
                "evaluation_seed": self.seed + 10_000, **asdict(stats)})
        save_model(self.model, self.files.last_model)
        print(f"평가 | steps={self.model.num_timesteps} episodes={self.completed} "
              f"return={stats.mean_reward:.2f} 생존={stats.success_rate:.0%} "
              f"정지={stats.settled_rate:.0%}", flush=True)

    def _on_training_start(self):
        self.evaluate()
        self.export(gif=True)
        if self.show_report:
            open_report(self.files)
        if self.preview:
            self.preview.start()
            self.preview.publish(self.model, self.completed, self.num_timesteps)

    def _on_step(self):
        for index, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                self.completed += 1
                episode = info["episode"]
                self.training_writer.writerow([self.completed, index, episode["r"], episode["l"],
                                               self.num_timesteps, time.perf_counter() - self.started])
                if self.completed == 1 or self.completed % 10 == 0:
                    print(f"episode={self.completed} steps={self.num_timesteps} return={episode['r']:.2f}", flush=True)
        evaluated = self.num_timesteps >= self.next_eval
        if evaluated:
            self.evaluate()
            self.next_eval = (self.num_timesteps // self.eval_interval + 1) * self.eval_interval
        gif_due = self.completed >= self.next_gif
        if gif_due:
            self.next_gif = (self.completed // self.gif_interval + 1) * self.gif_interval
        if evaluated or gif_due or time.perf_counter() - self.last_report >= self.report_interval:
            self.export(gif=gif_due)
        if self.preview and self.preview.due:
            self.training_file.flush()
            self.preview.publish(self.model, self.completed, self.num_timesteps)
        return True

    def finish(self, status):
        try:
            # SB3는 마지막 on_step 이후에도 gradient를 갱신하므로 최종 가중치를 다시 평가합니다.
            if status == "complete":
                self.evaluate()
            # 같은 에피소드 수여도 마지막 rollout 뒤 가중치가 달라질 수 있습니다.
            self.export(gif=True, status=status)
            if self.files.best_model.exists():
                self.export(gif=True, status=status, policy=load_policy(self.files.best_model), policy_label="best")
        finally:
            self.training_file.close()
            self.evaluation_file.close()
            if self.preview:
                self.preview.close()


def train(files: RunFiles, *, algorithm="sac", total_steps=100_000, num_envs=1, seed=42,
          learning_rate=.0003, gamma=.995, eval_interval=5000, eval_episodes=10,
          smoothing_alpha=.25, max_force_change=2., duration=10., goal="still",
          ppo_rollout_steps=256, gae_lambda=.95, log_std_init=-1., target_entropy=-2.,
          gif_interval=100, gif_fps=10, gif_duration=10., report_interval=5.,
          show_report=True, watch=False, watch_interval=5.):
    if algorithm not in ("sac", "ppo") or files.algorithm != algorithm:
        raise ValueError("algorithm and run folder must be the same sac/ppo")
    if min(total_steps, num_envs, eval_interval, eval_episodes, gif_interval) < 1 or seed < 0:
        raise ValueError("counts must be >= 1 and seed must be >= 0")
    if ppo_rollout_steps < 2 or not 0 <= gae_lambda <= 1:
        raise ValueError("PPO rollout must be >= 2 and gae_lambda in [0, 1]")
    if any(not math.isfinite(value) or value <= 0 for value in
           (learning_rate, report_interval, watch_interval, gif_duration)):
        raise ValueError("learning rate and time intervals must be finite and positive")
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if not math.isfinite(log_std_init) or not math.isfinite(target_entropy):
        raise ValueError("entropy/log_std settings must be finite")
    torch.set_num_threads(1)
    config = EnvConfig(duration_seconds=duration, reward_profile=goal)
    control = ContinuousControl(smoothing_alpha, max_force_change)
    reward = reward_settings(goal)
    factory = EnvFactory(config, control, reward)
    envs = (DummyVecEnv([factory]) if num_envs == 1 else
            SubprocVecEnv([factory] * num_envs, start_method="spawn"))
    model, callback, status = None, None, "failed"
    try:
        envs.seed(seed)
        if algorithm == "sac":
            model = SAC("MlpPolicy", envs, learning_rate=learning_rate, gamma=gamma,
                        buffer_size=200_000, learning_starts=1000, batch_size=128,
                        train_freq=4, gradient_steps=num_envs, ent_coef="auto_0.1",
                        target_entropy=target_entropy, policy_kwargs={"net_arch": [64, 64]},
                        device="cpu", seed=seed, verbose=0)
        else:
            model = PPO("MlpPolicy", envs, learning_rate=learning_rate, gamma=gamma,
                        n_steps=ppo_rollout_steps, batch_size=min(256, ppo_rollout_steps * num_envs),
                        n_epochs=10, gae_lambda=gae_lambda, ent_coef=0.,
                        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "log_std_init": log_std_init},
                        device="cpu", seed=seed, verbose=0)
        training = {"total_steps": total_steps, "num_envs": num_envs, "seed": seed,
            "learning_rate": learning_rate, "gamma": gamma, "eval_interval": eval_interval,
            "eval_episodes": eval_episodes, "evaluation_seed": seed + 10_000,
            "batch_size": model.batch_size}
        if algorithm == "sac":
            training.update({"buffer_size": 200_000, "learning_starts": 1000, "train_freq": 4,
                "gradient_steps": num_envs, "ent_coef": "auto_0.1", "target_entropy": target_entropy})
        else:
            training.update({"n_steps": ppo_rollout_steps, "n_epochs": 10,
                             "gae_lambda": gae_lambda, "log_std_init": log_std_init, "ent_coef": 0.})
        model.cartpole_settings = {"format_version": 1, "algorithm": algorithm.upper(),
            "environment": asdict(config), "control": asdict(control), "reward": reward,
            "observation_scale": [*OBSERVATION_SCALE[:4].tolist(), config.max_force],
            "training": training, "artifacts": {"gif_interval": gif_interval, "gif_fps": gif_fps,
                "gif_duration": gif_duration, "report_interval": report_interval,
                "preview_seed": seed + 20_000, "watch": watch, "watch_interval": watch_interval}}
        save_json(files.settings, model.cartpole_settings)
        callback = ResultCallback(files, config, seed=seed, eval_interval=eval_interval,
            eval_episodes=eval_episodes, gif_interval=gif_interval, gif_duration=gif_duration,
            gif_fps=gif_fps, report_interval=report_interval, show_report=show_report,
            watch=watch, watch_interval=watch_interval)
        print(f"{algorithm.upper()} | steps={total_steps} environments={num_envs} seed={seed}\n"
              f"결과: {files.root}\nworker_pids="
              f"{[p.pid for p in envs.processes] if num_envs > 1 else [os.getpid()]}", flush=True)
        model.learn(total_timesteps=total_steps, callback=callback)
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        print("학습 중단: 마지막 가중치를 저장합니다.", flush=True)
    finally:
        try:
            if model is not None:
                save_model(model, files.last_model)
            if callback is not None:
                callback.finish(status)
        finally:
            envs.close()
    print(f"결과 페이지: {files.report}\nBest: {files.best_model}\nLast: {files.last_model}", flush=True)
    return model
