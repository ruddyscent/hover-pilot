import os
import socket
import time
from typing import Mapping, Optional, Tuple

from hoverpilot.rflink.models import DEFAULT_CHANNEL_MAP, FlightAxisState, RFControlAction
from hoverpilot.rflink.protocol import (
    build_exchange_data_request,
    build_simple_request,
    parse_http_body,
    parse_state,
    state_looks_uninitialized,
)


class RFLinkConnectionError(ConnectionError):
    """Raised when the RealFlight Link TCP endpoint cannot be reached."""


class RFLinkStaleConnectionError(ConnectionError):
    """Raised when a previously healthy keep-alive socket was closed by RealFlight."""


class RFLinkClient:
    def __init__(
        self,
        host: str,
        port: int,
        channel_map: Optional[Mapping[str, int]] = None,
        socket_timeout_s: float = 1.0,
        request_attempts: int = 4,
        retry_backoff_s: float = 0.1,
        debug_state_flags: Optional[bool] = None,
    ):
        if socket_timeout_s <= 0.0:
            raise ValueError("socket_timeout_s must be greater than zero")
        if request_attempts < 1:
            raise ValueError("request_attempts must be at least 1")
        if retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s must be non-negative")
        self.host = host
        self.port = port
        self.channel_map = dict(DEFAULT_CHANNEL_MAP if channel_map is None else channel_map)
        self.socket_timeout_s = socket_timeout_s
        self.request_attempts = request_attempts
        self.retry_backoff_s = retry_backoff_s
        self.debug_state_flags = (
            _env_flag_enabled("RFLINK_DEBUG_STATE_FLAGS") if debug_state_flags is None else debug_state_flags
        )
        self.sock = None
        self._buffer = b""
        self._socket_response_count = 0
        self._peer_closes_connections = False
        self._controller_started = False
        self._printed_zero_state_debug = False
        self._last_flag_debug_tuple: Optional[Tuple[float, float, float]] = None

    def connect(self):
        last_error = None
        for attempt in range(1, self.request_attempts + 1):
            try:
                self._open_socket(log=attempt == 1)
                self._start_controller()
                return
            except (ConnectionError, OSError) as exc:
                last_error = exc
                self._reset_connection()
                if attempt < self.request_attempts:
                    self._wait_before_retry("connect", attempt, exc)
        raise RFLinkConnectionError(
            f"unable to initialize RealFlight Link at {self.host}:{self.port} "
            f"after {self.request_attempts} attempts with {self.socket_timeout_s:.1f}s timeout"
        ) from last_error

    def request_state(self, action: Optional[RFControlAction] = None) -> FlightAxisState:
        last_error = None
        response = None
        attempt = 1
        stale_rollovers = 0
        while attempt <= self.request_attempts:
            try:
                self._ensure_controller_ready()
                self._send_exchange_request(action)
                response = self._receive_http_response()
                break
            except RFLinkStaleConnectionError as exc:
                # An unexpected keep-alive loss can briefly hand control back to
                # the original device even though the injection command normally
                # survives planned short-lived transports. Re-inject before the
                # retry so no uncontrolled frame reaches the aircraft.
                last_error = exc
                stale_rollovers += 1
                self._peer_closes_connections = True
                self._reset_connection()
                if stale_rollovers >= self.request_attempts:
                    break
            except (ConnectionError, OSError) as exc:
                last_error = exc
                self._reset_connection()
                if attempt < self.request_attempts:
                    self._wait_before_retry("ExchangeData", attempt, exc)
                attempt += 1
        if response is None:
            raise RFLinkConnectionError(
                f"RealFlight Link ExchangeData failed at {self.host}:{self.port} "
                f"after {self.request_attempts} attempts"
            ) from last_error

        body = parse_http_body(response)
        state = parse_state(body)
        if state_looks_uninitialized(state) and not self._printed_zero_state_debug:
            self._printed_zero_state_debug = True
            print("[RFLINK] Received zeroed state. First SOAP body follows:")
            print(body[:2000])
        self._maybe_print_flag_debug(state)
        return state

    def step(self, action: Optional[RFControlAction] = None) -> FlightAxisState:
        return self.request_state(action=action)

    def close(self, restore_controller: bool = True):
        controller_started = self._controller_started
        self._close_socket()

        if restore_controller and controller_started:
            self._restore_original_controller()
        self._controller_started = False

    def _close_socket(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None
        self._buffer = b""
        self._socket_response_count = 0

    def _ensure_controller_ready(self):
        if self.sock is None:
            self._open_socket(log=False)
        if not self._controller_started:
            self._start_controller()

    def _open_socket(self, log: bool):
        # Opening a new TCP transport does not change RealFlight's injected
        # controller state. Keep those lifecycles independent.
        self._close_socket()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.socket_timeout_s)
        try:
            self.sock.connect((self.host, self.port))
        except (ConnectionError, OSError, TimeoutError, socket.timeout) as exc:
            try:
                self.sock.close()
            finally:
                self.sock = None
                self._buffer = b""
            raise RFLinkConnectionError(
                f"unable to connect to RealFlight Link at {self.host}:{self.port} "
                f"within {self.socket_timeout_s:.1f}s"
            ) from exc
        if log:
            print(f"[RFLINK] Connected to {self.host}:{self.port}")

    def _reset_connection(self, *, preserve_controller: bool = False):
        controller_started = self._controller_started
        self._close_socket()
        self._controller_started = controller_started if preserve_controller else False

    def _wait_before_retry(self, operation: str, attempt: int, exc: BaseException):
        delay_s = self.retry_backoff_s * (2 ** (attempt - 1))
        print(
            f"[RFLINK] {operation} attempt {attempt}/{self.request_attempts} failed: "
            f"{exc}; retrying in {delay_s:.2f}s"
        )
        if delay_s > 0.0:
            time.sleep(delay_s)

    def _start_controller(self):
        self._call_simple_action(
            "InjectUAVControllerInterface",
            "<InjectUAVControllerInterface><a>1</a><b>2</b></InjectUAVControllerInterface>",
        )
        if self.sock is None:
            self._open_socket(log=False)
        self._controller_started = True

    def _restore_original_controller(self):
        request_body = (
            "<RestoreOriginalControllerDevice><a>1</a><b>2</b></RestoreOriginalControllerDevice>"
        )
        for attempt in range(1, 4):
            restore_sock = None
            try:
                # Always restore on a fresh short-lived connection. By shutdown time the
                # long-lived ExchangeData socket may already be stale, and using it can
                # leave RealFlight's original InterLink controller un-restored.
                restore_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                restore_sock.settimeout(self.socket_timeout_s)
                restore_sock.connect((self.host, self.port))
                restore_sock.sendall(
                    build_simple_request(
                        self.host,
                        "RestoreOriginalControllerDevice",
                        request_body,
                    )
                )
                _receive_single_http_response(restore_sock)
                print(f"[RFLINK] RestoreOriginalControllerDevice succeeded on attempt {attempt}")
                return
            except (ConnectionError, OSError, TimeoutError, socket.timeout) as exc:
                if attempt == 3:
                    print(f"[RFLINK] RestoreOriginalControllerDevice failed: {exc}")
                else:
                    time.sleep(0.1)
            finally:
                if restore_sock is not None:
                    try:
                        restore_sock.close()
                    except Exception:
                        pass

    def _call_simple_action(self, action: str, body_inner_xml: str):
        if self.sock is None:
            self._open_socket(log=False)
        self.sock.sendall(build_simple_request(self.host, action, body_inner_xml))
        self._receive_http_response(close_after_read=True)

    def _send_exchange_request(self, action: Optional[RFControlAction] = None):
        self._ensure_socket()
        channel_values = None if action is None else action.to_channel_values(self.channel_map)
        try:
            self.sock.sendall(
                build_exchange_data_request(self.host, channel_values=channel_values)
            )
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
            if self._socket_response_count > 0:
                raise RFLinkStaleConnectionError(
                    "keep-alive connection closed before the next request"
                ) from exc
            raise

    def _receive_http_response(self, close_after_read: bool = False) -> bytes:
        self._ensure_socket()

        while b"\r\n\r\n" not in self._buffer:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout as exc:
                raise TimeoutError("timed out while receiving headers") from exc
            if not chunk:
                if self._socket_response_count > 0:
                    raise RFLinkStaleConnectionError(
                        "keep-alive connection closed before the next response"
                    )
                raise ConnectionError("connection closed while receiving headers")
            self._buffer += chunk

        headers, remainder = self._buffer.split(b"\r\n\r\n", 1)
        content_length = _parse_content_length(headers)

        while len(remainder) < content_length:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout as exc:
                raise TimeoutError("timed out while receiving body") from exc
            if not chunk:
                raise ConnectionError("connection closed while receiving body")
            remainder += chunk

        body = remainder[:content_length]
        self._buffer = remainder[content_length:]
        self._socket_response_count += 1
        if _header_requests_connection_close(headers):
            self._peer_closes_connections = True
        if close_after_read or self._peer_closes_connections:
            self._reset_connection(preserve_controller=self._controller_started)
        return headers + b"\r\n\r\n" + body


    def _maybe_print_flag_debug(self, state: FlightAxisState):
        if not self.debug_state_flags:
            return

        flag_tuple = (
            float(state.m_hasLostComponents),
            float(state.m_anEngineIsRunning),
            float(state.m_isTouchingGround),
        )
        if flag_tuple == self._last_flag_debug_tuple:
            return

        self._last_flag_debug_tuple = flag_tuple
        print(
            "[RFLINK:flags] "
            f"lost={flag_tuple[0]:0.1f} "
            f"engine={flag_tuple[1]:0.1f} "
            f"ground={flag_tuple[2]:0.1f}"
        )

    def _ensure_socket(self):
        if not self.sock:
            raise RuntimeError("socket is not connected")



def _parse_content_length(headers: bytes) -> int:
    for line in headers.decode("iso-8859-1").split("\r\n"):
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("missing Content-Length header")


def _header_requests_connection_close(headers: bytes) -> bool:
    for line in headers.decode("iso-8859-1").split("\r\n"):
        name, separator, value = line.partition(":")
        if separator and name.strip().lower() == "connection":
            tokens = {token.strip().lower() for token in value.split(",")}
            return "close" in tokens
    return False


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _receive_single_http_response(sock: socket.socket) -> bytes:
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        try:
            chunk = sock.recv(4096)
        except socket.timeout as exc:
            raise TimeoutError("timed out while receiving headers") from exc
        if not chunk:
            raise ConnectionError("connection closed while receiving headers")
        buffer += chunk

    headers, remainder = buffer.split(b"\r\n\r\n", 1)
    content_length = _parse_content_length(headers)

    while len(remainder) < content_length:
        try:
            chunk = sock.recv(4096)
        except socket.timeout as exc:
            raise TimeoutError("timed out while receiving body") from exc
        if not chunk:
            raise ConnectionError("connection closed while receiving body")
        remainder += chunk

    body = remainder[:content_length]
    return headers + b"\r\n\r\n" + body
