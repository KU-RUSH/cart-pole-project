from importlib import import_module
import time

from src.config import EnvConfig
from src.environment import CartPoleEnv


# 실행 옵션 (제어기, 렌더링, 실행 시간)
CONTROLLER = "lqr"  # "pid", "lqr", "my_controller", "dqn"
SHOW_VIEWER = True  # True: 시뮬레이션 render O / False: 시뮬레이션 render X
DURATION_SECONDS = 10.0

# 외란 옵션 (카트에 가해지는 외력)
USE_DISTURBANCE = True
DISTURBANCE_FORCE_N = 10.0
DISTURBANCE_START_SECONDS = 3.0
DISTURBANCE_DURATION_SECONDS = 0.25


def disturbance_at(sim_time: float) -> float:
    """현재 시각에 카트에 가할 외란 힘을 반환합니다."""
    if not USE_DISTURBANCE:
        return 0.0
    end_time = DISTURBANCE_START_SECONDS + DISTURBANCE_DURATION_SECONDS
    if DISTURBANCE_START_SECONDS <= sim_time < end_time:
        return DISTURBANCE_FORCE_N
    return 0.0


def run_simulation(controller_name="pid", render=False, duration=None):
    """선택한 제어기로 시뮬레이션을 실행합니다."""
    if DISTURBANCE_START_SECONDS < 0.0:
        raise ValueError("DISTURBANCE_START_SECONDS must be >= 0")
    if DISTURBANCE_DURATION_SECONDS < 0.0:
        raise ValueError("DISTURBANCE_DURATION_SECONDS must be >= 0")

    config = EnvConfig()
    if duration is not None:
        config = EnvConfig(duration_seconds=duration)

    # 예: controller_name="lqr"이면 src/controllers/lqr.py를 읽어서 실행함.
    # 같은 형식의 파일을 추가하면 별도의 등록 과정 없이 바로 실행할 수 있습니다.
    module_name = f"src.controllers.{controller_name}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise ValueError(f"src/controllers/{controller_name}.py 파일을 찾을 수 없습니다.") from error
        raise
    controller = module.make_controller(config.control_hz, config.max_force)

    # 제어기와 환경을 초기화합니다. with 블록이 끝나면 화면도 닫힙니다.
    with CartPoleEnv(config) as env:
        controller.reset()
        state = env.reset()
        if render and not env.render():
            raise KeyboardInterrupt
        started = time.perf_counter()

        print(f"controller={controller_name}")
        if USE_DISTURBANCE:
            print(
                f"disturbance={DISTURBANCE_FORCE_N:.2f}N  "
                f"start={DISTURBANCE_START_SECONDS:.2f}s  "
                f"duration={DISTURBANCE_DURATION_SECONDS:.2f}s"
            )

        # Control Core
        while True:
            force = controller.compute(state)
            disturbance = disturbance_at(state.sim_time)
            result = env.step(force, disturbance, stop_on_failure=False)
            state = result.state

            if render:
                if not env.render():
                    raise KeyboardInterrupt
                # 화면을 볼 때만 실제 시간에 맞춰 기다립니다.
                remaining = started + state.sim_time - time.perf_counter()
                time.sleep(max(0.0, remaining))

            if result.truncated:
                break

        print(f"time={state.sim_time:.2f}s")
        print("simulation finished")
        return state


def main():
    try:
        run_simulation(CONTROLLER, SHOW_VIEWER, DURATION_SECONDS)
    except KeyboardInterrupt:
        print("실행을 중단했습니다.")
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"오류: {error}") from error
    except NotImplementedError as error:
        raise SystemExit(f"DQN 스켈레톤이 아직 미완성입니다: {error}") from error


if __name__ == "__main__":
    main()
