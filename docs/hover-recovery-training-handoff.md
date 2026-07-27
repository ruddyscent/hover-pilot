# HoverPilot 회복 호버링 훈련 작업 기록

작성일: 2026-07-27 (Asia/Seoul)  
저장소: `/Users/kwchun/Workspace/hover-pilot`  
목적: Claude에 전달하여 기술 블로그 작성의 근거 자료로 사용

## 1. 결과 요약

Airplane Hover Trainer에서 에피소드가 시작된 뒤 일정 시간 동안 조종면을
중립으로 두고 쓰로틀만 유지하여 기체가 기울어진 다음, PPO 정책이 자세와
위치를 회복하도록 훈련했다.

최종 조건은 다음과 같다.

- episode-start idle 쓰로틀: `0.66`
- idle 커리큘럼: `0초 → 3초`
- 정책 인계 시간: `0.1초`
- 인계 보간: cubic smoothstep
- 인계 중 한 물리 스텝의 최대 액션 변화량: `0.25`
- 검증 에피소드 길이: `600` control steps
- 최종 검증: 첫 묶음 `3/3`, 추가 묶음 `5/5`, 합계 `8/8` 완주
- 최종 8회에서 지면 충돌: `0`
- 최종 8회에서 원통 경계 이탈: `0`
- 최종 코드 테스트: `191 passed`

최종 모델은 아래 파일이다.

```text
/Users/kwchun/Workspace/hover-pilot/ppo_hoverpilot_recovery_final.pt
```

이 파일은 `ppo_hoverpilot_recovery_curriculum_v19.pt`와 바이트 단위로
동일하다.

```text
SHA-256:
d38ab63815eb36552056d7a1a55e98f3fc93b8d65b35218282e7ed2a1624f2fb
```

## 2. 문제 정의

초기 호버 정책은 안정된 상태에서 제어를 시작하면 동작했지만, 제어 시작을
늦춰 기체가 기울고 이동한 뒤에는 회복 성능이 충분하지 않았다. 목표는 단순히
오래 떠 있는 것이 아니라 다음 순서를 반복해서 학습하는 것이었다.

1. Trainer reset으로 기체가 초기 호버 위치에 배치된다.
2. 조종면 `aileron/elevator/rudder=0`을 유지한다.
3. 쓰로틀만 `0.66`으로 고정한다.
4. 커리큘럼에 따라 `0~3초`를 기다린다.
5. 기울기와 위치 오차가 생긴 상태에서 정책에 제어를 넘긴다.
6. 기체가 trainer cylinder 안에서 안정적인 호버 상태로 복귀한다.

초기 idle 쓰로틀은 `0.55`, `0.60`을 거쳐 최종적으로 `0.66`을 사용했다.
낮은 값에서는 3초 idle 동안 고도 손실과 지면 접근 위험이 커졌다. 반대로
너무 높은 값은 수평 이동을 키워 원통 경계 충돌 가능성을 높이므로 `0.66`을
idle 전용 값으로 고정했다.

idle 쓰로틀 `0.66`과 정책의 호버 쓰로틀 trim은 서로 다른 값이다. 최종
체크포인트의 정책 쓰로틀 trim은 약 `0.750017`이다.

## 3. 구현 과정

### 3.1 물리시간 기반 idle

벽시계 시간이 아니라 RealFlight의 simulator physics time으로 idle 구간을
측정했다. 시뮬레이터나 통신 주기에 따라 실제 물리 스텝 수가 달라져도
항상 요청한 시간만큼 기체를 자유롭게 움직이게 하기 위해서다.

추가된 주요 CLI 옵션:

```text
--episode-start-idle-seconds
--episode-start-idle-throttle
--episode-start-idle-curriculum-steps
--episode-start-idle-curriculum-start-seconds
--episode-start-handoff-seconds
```

기본 idle 쓰로틀은 `0.66`이다. idle duration의 기본값은 `0`이므로 기존
즉시 제어 동작은 보존된다.

### 3.2 reset 경계 동기화

idle 훈련에서는 이전 추락 상태나 trainer reset 직전의 정지 상태를 새
에피소드로 잘못 받아들이면 안 된다. reset boundary를 명시적으로 기다리고,
trainer가 새 기체를 안정적으로 배치한 뒤에만 idle을 시작하도록 episode
lifecycle을 보강했다.

### 3.3 0초에서 3초까지의 커리큘럼

idle duration을 처음부터 3초로 두면 초기 정책이 대부분 곧바로 실패하여
학습 신호가 약해졌다. 따라서 `0초 → 3초` 선형 커리큘럼을 도입했다.

단순히 전체 학습 스텝으로 난도를 올리지 않고, 에피소드가 time limit까지
완주하고 마지막 상태가 위치·수평 속도·고도·기울기 기준을 만족했을 때만
커리큘럼 진행량을 늘린다. 실패 경험은 PPO rollout에는 남기되 난도 상승은
일시 정지한다.

### 3.4 idle에서 정책으로의 부드러운 인계

idle 종료 직후 정책 액션이 `0`에서 큰 값으로 바뀌며 기체가 튀는 현상이
관찰되었다. 다음 방식으로 완화했다.

- `0.1초` 동안 cubic smoothstep으로 idle 액션과 정책 액션을 보간
- 시뮬레이터의 매 물리 스텝마다 deterministic policy target을 다시 계산
- 한 물리 스텝의 액션 변화량을 최대 `0.25`로 제한
- 인계 구간은 일반 episode step limit에서 제외

변화량 제한을 `0.10`으로 더 낮추는 실험에서는 실제 인계가 약
`0.25~0.35초`로 길어졌고, 초기 롤 제동이 늦어져 시험한 `4/4` 에피소드가
모두 경계 이탈로 끝났다. 시각적인 부드러움만 높이고 회복 성능을 떨어뜨리는
설정이어서 폐기하고, 불연속 점프를 없애면서 제어 권한은 빠르게 넘기는
`0.25`를 유지했다.

### 3.5 회복 좌표계와 정책 표현력

수직 호버 근처에서는 RealFlight Euler roll 값이 특이점 부근에서
불연속적으로 바뀐다. 따라서 full-control 정책에서는 body roll rate를
적분한 연속적인 롤 오차를 사용했다. 기체가 롤하면서 elevator/rudder가
작용하는 수평축도 함께 회전하므로 위치와 속도 오차를 이 회전 좌표계로
투영했다.

회전 좌표계 부호를 반대로 적용한 A/B 검증은 `0/3`으로 더 나빠져 원래
부호로 되돌렸다.

all-controls actor는 검증된 구조화 축별 제어를 유지하면서, 구조화 정책으로
표현하지 못하는 결합 동역학을 학습할 수 있도록 scale `0.2`의 bounded
residual 경로를 추가했다. 기존 체크포인트에는 이 레이어가 없으므로 누락된
residual weight를 0으로 초기화하는 하위 호환 로딩도 구현했다.

수평 위치 회복을 강화하기 위해 standard reward의 position error weight를
`0.15 → 4.0`으로 조정했다.

## 4. 주요 실험과 판단

### 4.1 성능 진행

아래 수치는 RealFlight 실기 시뮬레이션 평가에서 얻었다.

| 단계 | 조건 | 결과 | 판단 |
|---|---|---:|---|
| 초기 고정 3초 | 이전 후보 | `0/5`, 평균 길이 약 `153` | 곧바로 3초는 너무 어려움 |
| curriculum v11 | 고정 2초 | `3/3`, 평균 위치 오차 `1.287 m` | 2초 회복 가능 |
| curriculum v14 | 고정 2.5초, recovery gain 2/3 | `3/3`, 평균 위치 오차 `2.338 m` | 후반 커리큘럼 기반으로 채택 |
| 긴 handoff | `0.3초`, `0.5초` | 각각 좋지 않음, 대표 평가 `0/3` | 회복 제어 지연 |
| throttle trim 0.70 | 고정 3초 | `2/3` 완주 | 고도에는 유리하지만 불충분 |
| throttle trim 0.75 후보 | 고정 3초 | `3/3`, 후속 `3/5` | 가장 유망한 영역 |
| v14 고정 3초 재평가 | trim 0.75 | `0/3` | 초기 방향에 따른 편차 큼 |
| v19 최종 후보 | trim 약 0.75, handoff 0.1초 | `3/3` + `5/5` | 최종 선택 |

고정 3초 상태에서 v14를 추가 미세학습하는 실험은 오히려 경계 이탈을
늘렸다. 저학습률 실험도 성능 저하 징후가 있어 중단했고, emergency
checkpoint `ppo_hoverpilot_recovery_final_v2.pt`는 최종 후보에서 제외했다.

### 4.2 최종 평가 수치

첫 번째 3회:

```text
avg_reward=-5311.882
avg_length=600.0
reward_per_step=-8.853
position_error=3.600 m
altitude_error=0.237 m
idle_end_tilt=11.890 deg
terminations={'truncated': 3}
```

추가 5회:

```text
avg_reward=-6159.745
avg_length=600.0
reward_per_step=-10.266
position_error=3.626 m
altitude_error=0.239 m
idle_end_tilt=12.478 deg
terminations={'truncated': 5}
```

여기서 `truncated`는 실패가 아니라 설정한 `600` step time limit까지
완주했다는 뜻이다. 두 평가를 합쳐 `8/8` 완주했고, failure termination은
없었다.

`attitude_error` 로그는 축별 적분 오차의 합을 포함해 일반적인 Euler
기울기와 직접 비교하기 어려우므로 블로그의 대표 성능 지표로 사용하지 않는
편이 좋다. 회복 성공률, episode length, position/altitude error,
idle-end tilt를 우선 사용한다.

## 5. 최종 사용 명령

### 재학습

```bash
cd /Users/kwchun/Workspace/hover-pilot

uv run hoverpilot-ppo train \
  --episode-start-idle-seconds 3.0 \
  --episode-start-idle-throttle 0.66 \
  --episode-start-idle-curriculum-steps 60000 \
  --episode-start-handoff-seconds 0.1 \
  --max-episode-steps 600 \
  --timesteps 120000 \
  --save-path ppo_hoverpilot_recovery.pt
```

이 명령은 현재 권장 재현 설정이다. 아래 TensorBoard precursor run의 일부는
탐색 과정에서 handoff `0.3초`, 다른 recovery gain 또는 다른 curriculum
길이를 사용했으므로 이벤트 파일의 `run/config/text_summary`를 함께 확인해야
한다.

### 최종 체크포인트 실행

```bash
cd /Users/kwchun/Workspace/hover-pilot

uv run hoverpilot-ppo play \
  --checkpoint ppo_hoverpilot_recovery_final.pt
```

현재 `play`는 `--max-episode-steps`를 생략하면 충돌이나 trainer boundary
exit 같은 실제 종료 조건이 발생할 때까지 한 에피소드를 무제한 실행한다.
프로세스는 `Ctrl+C`로 중단한다.

명시적으로 제한하려면:

```bash
uv run hoverpilot-ppo play \
  --checkpoint ppo_hoverpilot_recovery_final.pt \
  --episodes 5 \
  --max-episode-steps 600
```

## 6. Git 커밋

이번 작업과 직접 관련된 커밋:

| 커밋 | 전체 SHA | 내용 |
|---|---|---|
| `f490a96` | `f490a968af8d4dea7d432fff87fde1723ae35708` | 사용자 요청에 따라 기존 README를 작업 전에 먼저 커밋 |
| `91f8c53` | `91f8c53ad4b3fd747df008a946834eff7cf20f40` | physics-time idle, 0→3초 curriculum, smooth handoff, recovery frame, CLI, 로그와 테스트 |
| `84b8e28` | `84b8e28a806723318e3e22026c83d3aa2c66e8fe` | `play`에서 step limit 생략 시 에피소드를 무제한 실행 |

기반 구현:

| 커밋 | 전체 SHA | 내용 |
|---|---|---|
| `e0e48c7` | `e0e48c7c1622f9d7a7568262a53b63f88155c9d2` | structured all-controls PPO hover training 기반 |

관련 구현의 핵심 파일:

```text
README.md
src/hoverpilot/envs/hover_env.py
src/hoverpilot/rl/ppo.py
src/hoverpilot/training/hover.py
src/hoverpilot/utils/logger.py
tests/test_hover_env.py
tests/test_rl_ppo.py
```

## 7. TensorBoard 로그

실행:

```bash
cd /Users/kwchun/Workspace/hover-pilot
uv run tensorboard --logdir runs --port 6006
```

### 가장 관련성이 높은 이벤트 파일

#### 후반 recovery curriculum

```text
runs/hoverpilot-ppo-recovery-final-curriculum/events.out.tfevents.1785089241.MacBook-Pro-14.35444.0
```

- 파일 크기: `85,666 bytes`
- 최대 기록 step: `7,753`
- 14개 episode 기록
- 마지막 기록 episode length: `600`
- TensorBoard config:
  - idle throttle `0.66`
  - idle max `3.0초`
  - curriculum start `0.44초`
  - curriculum steps `30,000`
  - 이 precursor run의 handoff는 `0.3초`
  - max episode steps `600`
  - learning rate `3e-5`
  - policy std `0.03`

같은 디렉터리의 앞선 run:

```text
runs/hoverpilot-ppo-recovery-final-curriculum/events.out.tfevents.1785086842.MacBook-Pro-14.25840.0
```

#### 0→3초 curriculum 장기 run

```text
runs/hoverpilot-ppo-recovery-curriculum-0-to-3/events.out.tfevents.1785085595.MacBook-Pro-14.20953.0
```

- 파일 크기: `235,701 bytes`
- 최대 기록 step: `21,300`
- 마지막 PPO update step: `20,480`
- 마지막 평균 episode length: `900`
- idle `0→3초`, throttle `0.66`
- 이 탐색 run은 handoff `0.3초`, max episode `900`, recovery gain `3/10`

#### bounded residual 검증 run

```text
runs/hoverpilot-ppo-recovery-bounded-residual/events.out.tfevents.1785088352.MacBook-Pro-14.32184.0
```

- 파일 크기: `118,358 bytes`
- 최대 기록 step: `10,500`
- 마지막 평균 episode length: `600`
- structured all-controls actor에 residual 경로를 추가한 실험을 추적할 때 유용

#### 위치 좌표계 및 reset 관련 run

```text
runs/hoverpilot-ppo-recovery-position-aligned/events.out.tfevents.1785087343.MacBook-Pro-14.27843.0
runs/hoverpilot-ppo-recovery-position-aligned/events.out.tfevents.1785087554.MacBook-Pro-14.28668.0
runs/hoverpilot-ppo-recovery-position-aligned/events.out.tfevents.1785087750.MacBook-Pro-14.29470.0
runs/hoverpilot-ppo-recovery-reset-curriculum/events.out.tfevents.1785087942.MacBook-Pro-14.30284.0
runs/hoverpilot-ppo-recovery-reset-curriculum/events.out.tfevents.1785088037.MacBook-Pro-14.30726.0
```

#### handoff 비교 run

```text
runs/hoverpilot-ppo-recovery-handoff-1s/events.out.tfevents.1785086458.MacBook-Pro-14.24335.0
```

### 로그 해석 시 주의점

- 최종 v19 후보의 `3/3 + 5/5` 평가는 빠른 RealFlight 반복 검증을 위해
  `tensorboard_log_dir=None`으로 실행했다. 따라서 최종 `8/8` 수치는
  TensorBoard 이벤트 파일 안에 없고 이 문서의 콘솔 평가 기록을 근거로 한다.
- 위 이벤트 파일들은 최종 모델로 이어진 precursor와 ablation 기록이다.
  동일한 이름의 디렉터리라도 이벤트별 config가 다를 수 있으므로
  `run/config/text_summary`를 반드시 확인한다.
- 이벤트 파일은 `.gitignore`의 `runs/**/events.out.tfevents.*` 규칙에 의해
  Git에 포함되지 않는다. Claude에 별도 파일 또는 `runs` 디렉터리를 전달해야
  한다.

전체 recovery TensorBoard inventory는 다음 명령으로 얻을 수 있다.

```bash
find runs -type f -name 'events.out.tfevents.*' \
  -path '*recovery*' | sort
```

## 8. PyTorch 체크포인트

### 반드시 전달할 파일

| 용도 | 파일 | SHA-256 |
|---|---|---|
| 최종 배포/실행 | `ppo_hoverpilot_recovery_final.pt` | `d38ab63815eb36552056d7a1a55e98f3fc93b8d65b35218282e7ed2a1624f2fb` |
| 최종 후보 원본 | `ppo_hoverpilot_recovery_curriculum_v19.pt` | `d38ab63815eb36552056d7a1a55e98f3fc93b8d65b35218282e7ed2a1624f2fb` |

두 파일은 동일하며 크기는 각각 `85,275 bytes`다. 하나만 전달해도 된다.

최종 checkpoint properties:

```text
checkpoint format: HoverPilot structured policy v2
control_mode: all
policy_preset: none
policy throttle trim: 0.750017
policy action std: approximately 0.02 per axis
```

### 주요 계보 및 비교 파일

| 파일 | SHA-256 | 비고 |
|---|---|---|
| `ppo_hoverpilot_recovery.pt` | `6da712a9b7cefda8e73b3282c3d09f947efee532da773e8e8ef82a00c3c8f05e` | 초기 recovery 결과 |
| `ppo_hoverpilot_recovery_default_gains_probe.pt` | `ae06eede86fd81673851bebc6b573136674aba2f0c751a23510d32d2f0ffa5f6` | 후반 curriculum precursor |
| `ppo_hoverpilot_recovery_curriculum_v14.pt` | `cb9f41daa89852ecb431fe0f4c578f76c961ef761964ebf53079517239a89aa9` | 2.5초 조건에서 강했던 후보 |
| `ppo_hoverpilot_recovery_curriculum_v15.pt` | `0a6a25e8637e6607de2ad41fcdf40c5fc2183fc924ca620033e3eda8dfaba287` | curriculum 후속 |
| `ppo_hoverpilot_recovery_curriculum_v16.pt` | `a25c11eb672f25afd0c7b75352a5dce075bb9f2282921d9e06f594de634a64ce` | curriculum 후속 |
| `ppo_hoverpilot_recovery_curriculum_v17.pt` | `246fee933c2fc829f9f968f3ea2e2bfd1b868891e668cd28f689a28ad6344b72` | v19 직전 후보 |
| `ppo_hoverpilot_recovery_final_v2.pt` | `aa9d78646d98961ef4ca755049b55778c4b47431be25b741d472cf8352b11a01` | 고정 3초 추가 미세학습 중 생성된 emergency checkpoint; 사용하지 말 것 |

그 밖의 `ppo_hoverpilot_recovery_*.pt` 파일은 gain, damping, lateral prior,
handoff, residual 등을 비교한 ablation checkpoint다. 파일명이 실험 목적을
나타낸다. 예:

```text
ppo_hoverpilot_recovery_pitch_gain*.pt
ppo_hoverpilot_recovery_rudder_velocity5.pt
ppo_hoverpilot_recovery_aileron_*.pt
ppo_hoverpilot_recovery_eval_*.pt
ppo_hoverpilot_recovery_gentle_*.pt
ppo_hoverpilot_recovery_damped*.pt
```

모든 `.pt` 파일은 `.gitignore`의 `*.pt` 규칙에 의해 Git에 포함되지 않는다.
Claude에 전달할 때는 최종 `.pt` 파일을 별도 첨부해야 한다.

체크포인트는 정책/value weight와 observation configuration을 담지만 optimizer
state나 curriculum exposure progress 전체를 복원하는 학습 snapshot은 아니다.
이어 학습할 때는 `--resume-from`과 함께 curriculum start seconds 및 나머지
하이퍼파라미터를 명시해야 한다.

## 9. 테스트와 성능 회귀 검토

recovery curriculum 구현 직후 전체 테스트:

```text
190 passed
```

`play`의 기본 episode step limit를 제거하고 테스트를 하나 추가한 최종 상태:

```text
191 passed in 4.59s
```

최종 정리 과정에서 임시 학습·평가 Python script는 모두 삭제했다. 제품 코드에
남긴 요소는 물리시간 idle, curriculum, reset boundary, smooth handoff,
회복 좌표계, structured residual, checkpoint compatibility, CLI, telemetry와
테스트다.

성능 저하를 일으킨 다음 후보는 최종 구현에서 제외했다.

- handoff max action step `0.10`
- 회전 recovery frame의 반대 부호
- handoff `0.3초`/`0.5초`의 최종 기본값 사용
- 고정 3초 상태에서의 추가 저학습률 미세학습
- final emergency checkpoint v2

## 10. Claude/블로그 작성자를 위한 권장 서사

1. 안정된 호버만 학습한 정책은 실제 회복 능력이 약했다.
2. 단순히 초기 상태를 랜덤화하는 대신, 실제 시뮬레이터에서 쓰로틀만 유지해
   자연스러운 기울기·속도·위치 오차를 만들었다.
3. 처음부터 3초로 시작하면 실패 경험만 쌓이므로 성공 기반 0→3초
   커리큘럼을 만들었다.
4. idle→policy 경계 자체가 새로운 불안정 요인이어서 physics-time
   smoothstep handoff를 추가했다.
5. 더 부드러운 제어가 항상 더 좋은 것은 아니었다. 변화량 제한 `0.10`은
   회복 제동을 늦춰 실패했고, `0.25`가 실제 성능과 시각적 부드러움의
   균형점이었다.
6. 수직 호버의 Euler singularity 때문에 body-rate 적분 좌표계가 필요했다.
7. 최종 후보는 3초 idle 후 평균 약 12도 기울어진 상태에서 제어를 시작해
   `8/8` 에피소드를 600 step 완주했다.

블로그에는 “완벽한 정밀 위치 호버”보다는 “의도적으로 악화시킨 초기
상태에서 trainer boundary 안으로 회복하고 장시간 유지하는 능력을
확보했다”라고 표현하는 것이 수치에 가장 충실하다.

