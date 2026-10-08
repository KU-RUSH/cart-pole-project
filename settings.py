"""실습 설정. 값을 수정하고 python train.py를 실행하세요."""

ALGORITHM = "dqn"             # "dqn", "sac", "ppo"
REWARD_PROFILE = "still"       # 보상 식과 가중치는 src/reward.py
EPISODES = 1200                # DQN: 모든 환경을 합친 완료 에피소드 수
TOTAL_STEPS = 100_000          # SAC/PPO: 모든 환경을 합친 환경 전이 수
NUM_ENVS = 1                  # 2 이상이면 여러 프로세스에서 경험 수집
SEED = 42
DURATION_SECONDS = 10.0

DQN = {
    "learning_rate": 0.0005,
    "gamma": 0.995,
    "epsilon_decay_steps": 50_000,
    "epsilon_min": 0.02,
    "eval_interval": 100,     # 완료 에피소드 수
}
CONTINUOUS = {
    "learning_rate": 0.0003,
    "gamma": 0.995,
    "eval_interval": 5_000,   # 환경 전이 수
    "ppo_rollout_steps": 256,
    "gae_lambda": 0.95,
    "log_std_init": -1.0,
    "target_entropy": -2.0,   # SAC
}

SMOOTHING_ALPHA = 0.25         # 목표 힘 쪽으로 이동할 비율
MAX_FORCE_CHANGE = 2.0        # 제어 한 번당 힘 변화 상한, N
EVAL_EPISODES = 10            # 학습 중 같은 초기 상태로 비교
GIF_INTERVAL = 100            # 완료 에피소드마다 최신 정책 GIF 저장
GIF_FPS = 10
GIF_DURATION_SECONDS = 10.0
REPORT_INTERVAL_SECONDS = 5.0 # 그래프 저장 간격, 실제 시간
OPEN_REPORT = True            # 그래프와 GIF가 있는 로컬 결과 페이지
WATCH = False                 # 별도 MuJoCo 창으로 최신 정책 보기
WATCH_INTERVAL_SECONDS = 5.0

EVALUATION_EPISODES = 100      # python results.py evaluate
EVALUATION_SEED = 30_000       # 학습 중 평가와 다른 초기 상태
