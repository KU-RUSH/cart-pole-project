# 카트폴 제어 예제

MuJoCo 카트폴을 PID와 LQR로 제어해 보고, 같은 환경에서 DQN을 직접 완성해 보는 작은 프로젝트입니다.

## 바로 실행하기

Python 3.11 이상에서 처음 한 번만 라이브러리를 설치합니다. VS Code 터미널이나
Anaconda 터미널에서 프로젝트 폴더로 이동한 뒤 실행하세요.

```bash
python -m pip install -r requirements.txt
```

그다음 `main.py`를 열고 파일 위쪽의 실행값을 원하는 대로 바꿉니다.

```python
CONTROLLER = "pid"       # "pid", "lqr", "my_controller", "dqn"
SHOW_VIEWER = True       # False이면 화면 없이 빠르게 실행
DURATION_SECONDS = 100.0 # 시뮬레이션 시간

USE_DISTURBANCE = False
DISTURBANCE_FORCE_N = 5.0
DISTURBANCE_START_SECONDS = 3.0
DISTURBANCE_DURATION_SECONDS = 0.25
```

VS Code 오른쪽 위의 **Python 파일 실행** 단추를 누르면 됩니다. 터미널에서
실행할 때도 별도 옵션 없이 `python main.py`만 입력합니다.

Windows에서 프로젝트의 가상환경을 사용하려면 VS Code의 Python 인터프리터를
`.venv`로 선택하거나 `python` 대신 `.\.venv\Scripts\python.exe`를 사용할 수
있습니다. macOS에서 MuJoCo 화면을 띄울 때는 `mjpython main.py`를 사용하세요.

초기 상태는 실행할 때마다 수직 근처에서 무작위로 정합니다. 지금은 시드를
설정하거나 관리하지 않습니다.

## 파일 구성

```text
main.py                     제어기와 시뮬레이터 실행
train.py                    DQN 학습 전체 반복
src/
  config.py                 실험 시간, 힘 제한, 초기 상태 범위
  state.py                  제어기에 전달되는 상태
  environment.py            MuJoCo 물리와 종료 조건
  reward.py                 학생이 설계할 DQN 보상 함수
  dqn_episode.py            DQN 에피소드 한 번의 진행 과정
  cartpole.xml              카트폴 물리 모델
  controllers/
    pid.py                  실행 가능한 PID 예제
    lqr.py                  실행 가능한 LQR 예제
    dqn.py                  직접 완성할 DQN 스켈레톤
    my_controller.py        자유롭게 수정할 제어기 예제
```

처음에는 `main.py`의 아래 세 줄을 중심으로 읽으면 됩니다.

```python
force = controller.compute(state)
disturbance = disturbance_at(state.sim_time)
result = env.step(force, disturbance, stop_on_failure=False)
state = result.state
```

상태는 `[x, x_dot, theta, theta_dot]`, 즉 카트 위치, 카트 속도, 막대 각도,
막대 각속도 순서입니다. `theta=0`은 막대가 수직으로 선 상태이며, 힘의
단위는 N입니다.

## PID와 LQR

PID 이득은 `src/controllers/pid.py`의 `make_controller()`에서 바꿀 수
있습니다. LQR의 상태 비용 Q와 힘 비용 R은
`src/controllers/lqr.py`의 `make_controller()`에서 바꿀 수 있습니다.

```python
# main.py
CONTROLLER = "lqr"
SHOW_VIEWER = True
DURATION_SECONDS = 5.0
```

`main.py`는 보상이나 점수를 계산하지 않습니다. 막대가 크게 기울어도 정한
시간까지 물리 시뮬레이션을 계속하므로 제어기와 외란의 반응만 확인할 수
있습니다.

직접 만든 제어기도 `reset()`과 `compute(state)`를 제공하고 파일 아래에
`make_controller(control_hz, max_force)`를 두면 같은 방식으로 실행됩니다.

```python
# main.py
CONTROLLER = "my_controller"
```

## 보상 설계

물리 환경에는 보상이 들어 있지 않습니다. 학생은 `src/reward.py`의
`calculate_reward()`만 수정해서 학습 목표를 바꿀 수 있습니다.

```python
def calculate_reward(state, force, next_state, terminated, config):
    # 현재 상태, 사용한 힘, 다음 상태와 실패 여부를 이용해 직접 설계
    return reward
```

따라서 PID/LQR 시뮬레이션은 보상 설계의 영향을 받지 않고, DQN 학습만 이
함수의 반환값을 사용합니다.

## DQN 스켈레톤

DQN 학습은 시뮬레이터 실행과 분리되어 있습니다.

- `src/reward.py`: 학생이 직접 보상 설계
- `src/dqn_episode.py`: 에피소드 한 번 실행
- `train.py`: 여러 에피소드를 반복하고 모델 저장

`src/controllers/dqn.py`에는 다음 구성만 준비되어 있습니다.

- 상태 정규화
- replay buffer
- Q-network 인터페이스
- DQN agent 인터페이스
- 학습한 모델을 실행 제어기로 연결하는 형식

파일의 `TODO 1 → TODO 2 → TODO 3` 순서로 신경망, epsilon-greedy 행동
선택, DQN 업데이트를 구현하세요. 한 에피소드의
**행동 선택 → 환경 진행 → 보상 계산 → 경험 저장 → 학습** 흐름은
`src/dqn_episode.py`에 모아 두었습니다.

```python
# train.py 위쪽
EPISODES = 300
OUTPUT_PATH = DEFAULT_CHECKPOINT
```

TODO를 완성한 다음 `train.py`를 VS Code 실행 단추로 실행합니다. 학습이
끝나면 `main.py`의 `CONTROLLER`를 `"dqn"`으로 바꾸어 결과를 확인합니다.
터미널을 쓸 때도 각각 `python train.py`, `python main.py`만 실행하면 됩니다.

DQN의 TODO를 구현하기 전에는 `train.py`가 무엇을 먼저 구현해야 하는지
짧은 안내를 출력하고 종료합니다. 기본 모델 저장 위치는
`results/dqn.npz`입니다.
