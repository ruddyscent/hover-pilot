# RealFlight Link Interface

`HoverPilot`에서 사용하는 RealFlight Link 인터페이스를 정리한 문서다. 이 문서는 현재 구현 기준으로 [`src/hoverpilot/rflink/client.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/client.py), [`src/hoverpilot/rflink/protocol.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/protocol.py), [`src/hoverpilot/rflink/models.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/models.py) 에 정의된 계약을 설명한다.

## 개요

RealFlight Link 연결은 TCP 기반 SOAP over HTTP 형태로 동작한다.

- 기본 포트: `18083`
- 연결 방식: `socket.AF_INET`, `SOCK_STREAM`
- 요청 형태: `POST / HTTP/1.1`
- 콘텐츠 타입: `text/xml;charset='UTF-8'`
- 연결 정책: `Connection: Keep-Alive` 요청을 기본으로 사용하되, RealFlight가
  `Connection: close`로 응답하거나 완료된 응답 뒤 socket을 닫으면 controller
  상태를 유지한 채 다음 `ExchangeData` 요청부터 새 TCP 연결을 사용한다.

클라이언트는 아래 순서로 동작한다.

1. RealFlight Link 소켓 연결
2. `InjectUAVControllerInterface` 호출로 외부 제어 인터페이스 활성화
3. `ExchangeData` 요청으로 RC 입력 전송 및 시뮬레이터 상태 수신
4. 종료 시 `RestoreOriginalControllerDevice` 호출로 원래 컨트롤러 복원

## Python 인터페이스

### `RFLinkClient`

위치: [`src/hoverpilot/rflink/client.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/client.py)

RealFlight Link와의 연결, 요청 전송, 응답 수신을 담당하는 상위 클라이언트다.

#### 생성자

```python
RFLinkClient(
    host: str,
    port: int,
    channel_map: Mapping[str, int] | None = None,
    socket_timeout_s: float = 1.0,
    debug_state_flags: bool | None = None,
)
```

파라미터:

- `host`: RealFlight Link 서버 주소
- `port`: RealFlight Link TCP 포트
- `channel_map`: 논리 조종축과 RealFlight 채널 인덱스 매핑
- `socket_timeout_s`: 소켓 송수신 타임아웃
- `debug_state_flags`: 상태 플래그 디버그 출력 활성화 여부

#### 주요 메서드

- `connect() -> None`
  - 소켓을 열고 `InjectUAVControllerInterface`를 호출한다.
- `request_state(action: RFControlAction | None = None) -> FlightAxisState`
  - RC 입력을 함께 보내고 최신 상태를 반환한다.
  - 연결 오류나 타임아웃 발생 시 연결을 재수립한 뒤 한 번 더 재시도한다.
- `step(action: RFControlAction | None = None) -> FlightAxisState`
  - `request_state()`의 별칭이다.
- `close(restore_controller: bool = True) -> None`
  - 소켓을 닫고, 기본값으로 원래 컨트롤러를 복원한다.

### `RFControlAction`

위치: [`src/hoverpilot/rflink/models.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/models.py)

RealFlight로 보낼 조종 입력을 표현한다.

```python
@dataclass(slots=True)
class RFControlAction:
    throttle: float = 0.0
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    channel_overrides: dict[int, float] = field(default_factory=dict)
```

입력 규칙:

- `aileron`, `elevator`, `rudder`: `[-1.0, 1.0]`
- `throttle`: `[0.0, 1.0]`
- `channel_overrides`: 각 채널별 `[0.0, 1.0]`
- 유한하지 않은 값(`NaN`, `inf`)은 `ValueError`
- 범위를 벗어난 값은 클램프된다

보조 메서드:

- `RFControlAction.neutral()`
- `RFControlAction.safe_idle()`
- `to_channel_values(channel_map=None) -> list[float]`

#### 기본 채널 매핑

```python
DEFAULT_CHANNEL_MAP = {
    "aileron": 0,
    "elevator": 1,
    "throttle": 2,
    "rudder": 3,
}
```

`to_channel_values()`는 항상 길이 `12`의 리스트를 반환한다.

- 양방향 조종축: `[-1, 1]` 값을 `[0, 1]`로 변환
- 스로틀: 그대로 `[0, 1]`
- 나머지 채널: 기본값 `0.0`
- `channel_overrides`가 있으면 마지막에 해당 채널 값을 덮어쓴다

예시:

```python
action = RFControlAction(
    throttle=0.55,
    aileron=0.0,
    elevator=0.0,
    rudder=0.0,
)

channel_values = action.to_channel_values()
# [0.5, 0.5, 0.55, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### `FlightAxisState`

위치: [`src/hoverpilot/rflink/models.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/models.py)

RealFlight Link의 `ExchangeData` 응답을 파싱한 상태 모델이다.

- `rcin`: 입력 채널 값 12개
- 나머지 필드: 비행 상태, 위치, 속도, 가속도, 배터리, 플래그 등
- `summary() -> str`: 주요 상태를 짧은 문자열로 요약

## 프로토콜 인터페이스

### SOAP Envelope

모든 요청은 SOAP Envelope 안에 XML 바디를 넣어 전송한다.

```xml
<?xml version='1.0' encoding='UTF-8'?>
<soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/' xmlns:xsd='http://www.w3.org/2001/XMLSchema' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
  <soap:Body>
    ...
  </soap:Body>
</soap:Envelope>
```

### HTTP 요청 헤더

프로토콜 생성 함수는 아래 헤더를 사용한다.

```http
POST / HTTP/1.1
Host: <host>
soapaction: '<action>'
content-length: <bytes>
content-type: text/xml;charset='UTF-8'
Connection: Keep-Alive
```

관련 함수:

- `build_soap_request(host, action, body) -> bytes`
- `build_simple_request(host, action, body_inner_xml) -> bytes`
- `build_exchange_data_request(host, channel_values=None, selected_channels=4095) -> bytes`

### `InjectUAVControllerInterface`

외부 제어 인터페이스를 RealFlight에 주입한다.

```xml
<InjectUAVControllerInterface>
  <a>1</a>
  <b>2</b>
</InjectUAVControllerInterface>
```

현재 클라이언트는 연결 직후 이 액션을 자동 호출한다.

### `ExchangeData`

조종 입력과 상태 교환에 사용하는 핵심 요청이다.

```xml
<ExchangeData>
  <pControlInputs>
    <m-selectedChannels>4095</m-selectedChannels>
    <m-channelValues-0to1>
      <item>...</item>
      ...
    </m-channelValues-0to1>
  </pControlInputs>
</ExchangeData>
```

필드 의미:

- `m-selectedChannels`
  - 기본값은 `4095`
  - 현재 구현에서는 12채널 모두 활성화하는 비트마스크로 사용한다
- `m-channelValues-0to1`
  - 길이 `12`의 `0.0 ~ 1.0` 실수 배열

### `RestoreOriginalControllerDevice`

종료 시 원래 컨트롤러를 복원한다.

```xml
<RestoreOriginalControllerDevice>
  <a>1</a>
  <b>2</b>
</RestoreOriginalControllerDevice>
```

현재 구현은 기존 장기 연결 대신, 짧은 새 연결을 열어 최대 3회 재시도한다.

## 응답 파싱 계약

관련 함수:

- `parse_http_body(data: bytes) -> str`
- `parse_state(xml_text: str) -> FlightAxisState`
- `state_looks_uninitialized(state: FlightAxisState) -> bool`

파싱 규칙:

- HTTP 응답은 `Content-Length` 기준으로 읽는다
- SOAP/HTTP 바디에서 실수로 파싱 가능한 텍스트만 수집한다
- `<item>` 태그 값은 입력 채널 상태 `rcin`으로 누적한다
- 알려진 XML 태그는 `STATE_FIELD_MAP`을 통해 Python 필드명으로 매핑한다
- 누락된 값은 기본값 `0.0`을 유지한다

### 미초기화 상태 판정

아래 필드가 모두 `0`에 가까우면 미초기화 상태로 본다.

- `m_currentPhysicsTime_SEC`
- `m_altitudeASL_MTR`
- `m_groundspeed_MPS`
- `m_airspeed_MPS`
- `m_orientationQuaternion_W`
- `m_flightAxisControllerIsActive`

이 판정은 클라이언트가 첫 수신 상태가 비정상적으로 전부 0인 경우를 감지하는 데 사용된다.

## 상태 필드 목록

아래는 `ExchangeData` 응답에서 `FlightAxisState`로 매핑되는 필드다.

### 입력 채널

- `rcin: list[float]`
  - `<item>` 태그에서 추출한 길이 12 배열

### 속도 및 고도

- `m_airspeed_MPS`
- `m_altitudeASL_MTR`
- `m_altitudeAGL_MTR`
- `m_groundspeed_MPS`

### 각속도 및 자세

- `m_pitchRate_DEGpSEC`
- `m_rollRate_DEGpSEC`
- `m_yawRate_DEGpSEC`
- `m_azimuth_DEG`
- `m_inclination_DEG`
- `m_roll_DEG`

### 위치

- `m_aircraftPositionX_MTR`
- `m_aircraftPositionY_MTR`

### 월드 좌표계 속도

- `m_velocityWorldU_MPS`
- `m_velocityWorldV_MPS`
- `m_velocityWorldW_MPS`

### 기체 좌표계 속도

- `m_velocityBodyU_MPS`
- `m_velocityBodyV_MPS`
- `m_velocityBodyW_MPS`

### 월드 좌표계 가속도

- `m_accelerationWorldAX_MPS2`
- `m_accelerationWorldAY_MPS2`
- `m_accelerationWorldAZ_MPS2`

### 기체 좌표계 가속도

- `m_accelerationBodyAX_MPS2`
- `m_accelerationBodyAY_MPS2`
- `m_accelerationBodyAZ_MPS2`

### 바람 및 동력

- `m_windX_MPS`
- `m_windY_MPS`
- `m_windZ_MPS`
- `m_propRPM`
- `m_heliMainRotorRPM`

### 전원 및 연료

- `m_batteryVoltage_VOLTS`
- `m_batteryCurrentDraw_AMPS`
- `m_batteryRemainingCapacity_MAH`
- `m_fuelRemaining_OZ`

### 상태 플래그

- `m_isLocked`
- `m_hasLostComponents`
- `m_anEngineIsRunning`
- `m_isTouchingGround`
- `m_currentAircraftStatus`
- `m_currentPhysicsTime_SEC`
- `m_currentPhysicsSpeedMultiplier`
- `m_flightAxisControllerIsActive`
- `m_resetButtonHasBeenPressed`

### 자세 쿼터니언

- `m_orientationQuaternion_X`
- `m_orientationQuaternion_Y`
- `m_orientationQuaternion_Z`
- `m_orientationQuaternion_W`

## 사용 예시

```python
from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.rflink.models import RFControlAction

client = RFLinkClient(host="127.0.0.1", port=18083)
client.connect()

action = RFControlAction(
    throttle=0.55,
    aileron=0.0,
    elevator=0.0,
    rudder=0.0,
)

state = client.step(action)
print(state.summary())

client.close()
```

## 구현상 주의사항

- `RFLinkClient.request_state()`는 연결 끊김, 타임아웃, 소켓 오류를 만나면 연결을 재설정하고 한 번 재시도한다.
- `close()`는 기본적으로 원래 컨트롤러를 복원한다.
- 상태 플래그 디버그 로그는 `RFLINK_DEBUG_STATE_FLAGS=1` 환경 변수로 켤 수 있다.
- RealFlight Link 상태 값은 트레이너 모드마다 의미가 조금 다를 수 있으므로, 일부 필드는 환경 로직에서 보수적으로 해석한다.

## 관련 파일

- [`README.md`](/Users/kwchun/Workspace/hover-pilot/README.md)
- [`src/hoverpilot/rflink/client.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/client.py)
- [`src/hoverpilot/rflink/protocol.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/protocol.py)
- [`src/hoverpilot/rflink/models.py`](/Users/kwchun/Workspace/hover-pilot/src/hoverpilot/rflink/models.py)
- [`tests/test_rflink_actions.py`](/Users/kwchun/Workspace/hover-pilot/tests/test_rflink_actions.py)
