# 카트폴 강화학습 예제

프로젝트 폴더의 PowerShell에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
.\.venv\Scripts\python.exe train.py             # 학습
.\.venv\Scripts\python.exe main.py              # 학습한 최고 모델 보기
.\.venv\Scripts\python.exe results.py           # 그래프·GIF 리포트 열기
.\.venv\Scripts\python.exe results.py evaluate  # 별도 초기 상태로 평가
```

처음에는 `train.py`부터 실행합니다. 학습 설정은 `settings.py`, 보상은
`src/reward.py`에서 수정합니다. 결과는 실행마다 `results/runs/`에 저장되며,
학습 중에도 리포트에서 그래프와 주기적인 GIF를 볼 수 있습니다.

설정 변경, 결과 해석, 코드 구조는 [실습 가이드](docs/GUIDE.md)를 참고하세요.
