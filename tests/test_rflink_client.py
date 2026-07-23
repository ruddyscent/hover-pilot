import socket
import unittest
from unittest.mock import Mock, patch

from hoverpilot.rflink.client import (
    RFLinkClient,
    RFLinkConnectionError,
    RFLinkStaleConnectionError,
)


class FailingSocket:
    def __init__(self, exc):
        self.exc = exc
        self.closed = False
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        raise self.exc

    def close(self):
        self.closed = True


class ReceivingSocket:
    def __init__(self, *chunks):
        self.chunks = list(chunks)
        self.closed = False
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.address = address

    def recv(self, size):
        del size
        return self.chunks.pop(0) if self.chunks else b""

    def close(self):
        self.closed = True


class RFLinkClientTests(unittest.TestCase):
    def test_connect_timeout_raises_clear_connection_error_and_closes_socket(self):
        fake_socket = FailingSocket(socket.timeout("timed out"))
        client = RFLinkClient(
            "10.0.0.2",
            18083,
            socket_timeout_s=0.2,
            retry_backoff_s=0.0,
        )

        with patch("hoverpilot.rflink.client.socket.socket", return_value=fake_socket):
            with self.assertRaises(RFLinkConnectionError) as context:
                client.connect()

        self.assertIn("10.0.0.2:18083", str(context.exception))
        self.assertIn("0.2s", str(context.exception))
        self.assertTrue(fake_socket.closed)
        self.assertIsNone(client.sock)

    def test_request_state_retries_transient_failures_with_exponential_backoff(self):
        client = RFLinkClient(
            "127.0.0.1",
            18083,
            request_attempts=3,
            retry_backoff_s=0.1,
        )
        expected_state = Mock()
        client._ensure_controller_ready = Mock()
        client._send_exchange_request = Mock()
        client._receive_http_response = Mock(
            side_effect=[
                ConnectionError("connection closed"),
                TimeoutError("server stalled"),
                b"response",
            ]
        )
        client._reset_connection = Mock()

        with (
            patch("hoverpilot.rflink.client.time.sleep") as sleep,
            patch("hoverpilot.rflink.client.parse_http_body", return_value=b"body"),
            patch("hoverpilot.rflink.client.parse_state", return_value=expected_state),
            patch("hoverpilot.rflink.client.state_looks_uninitialized", return_value=False),
        ):
            state = client.request_state()

        self.assertIs(state, expected_state)
        self.assertEqual(client._receive_http_response.call_count, 3)
        self.assertEqual(client._reset_connection.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.1, 0.2])

    def test_request_state_raises_typed_error_after_retry_budget_is_exhausted(self):
        client = RFLinkClient(
            "127.0.0.1",
            18083,
            request_attempts=3,
            retry_backoff_s=0.0,
        )
        client._ensure_controller_ready = Mock()
        client._send_exchange_request = Mock()
        client._receive_http_response = Mock(side_effect=TimeoutError("server stalled"))
        client._reset_connection = Mock()

        with self.assertRaisesRegex(RFLinkConnectionError, "after 3 attempts"):
            client.request_state()

        self.assertEqual(client._receive_http_response.call_count, 3)
        self.assertEqual(client._reset_connection.call_count, 3)

    def test_stale_keep_alive_reconnects_immediately_and_marks_controller_for_reinjection(self):
        client = RFLinkClient("127.0.0.1", 18083, request_attempts=3)
        expected_state = Mock()
        client._controller_started = True
        client._ensure_controller_ready = Mock()
        client._send_exchange_request = Mock()
        client._receive_http_response = Mock(
            side_effect=[
                RFLinkStaleConnectionError("peer closed keep-alive"),
                b"response",
            ]
        )
        client._reset_connection = Mock()
        client._wait_before_retry = Mock()

        with (
            patch("hoverpilot.rflink.client.parse_http_body", return_value=b"body"),
            patch("hoverpilot.rflink.client.parse_state", return_value=expected_state),
            patch("hoverpilot.rflink.client.state_looks_uninitialized", return_value=False),
        ):
            state = client.request_state()

        self.assertIs(state, expected_state)
        client._reset_connection.assert_called_once_with()
        self.assertEqual(client._ensure_controller_ready.call_count, 2)
        client._wait_before_retry.assert_not_called()
        self.assertTrue(client._peer_closes_connections)

    def test_reset_connection_clears_controller_state_by_default(self):
        client = RFLinkClient("127.0.0.1", 18083)
        client.sock = Mock()
        client._controller_started = True

        client._reset_connection()

        self.assertFalse(client._controller_started)
        self.assertIsNone(client.sock)

    def test_connection_close_response_switches_to_fresh_transport_mode(self):
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 3\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            b"abc"
        )
        sock = ReceivingSocket(response)
        client = RFLinkClient("127.0.0.1", 18083)
        client.sock = sock
        client._controller_started = True

        received = client._receive_http_response()

        self.assertEqual(received, response)
        self.assertTrue(sock.closed)
        self.assertIsNone(client.sock)
        self.assertTrue(client._controller_started)
        self.assertTrue(client._peer_closes_connections)

    def test_closed_reused_socket_is_classified_as_stale_keep_alive(self):
        client = RFLinkClient("127.0.0.1", 18083)
        client.sock = ReceivingSocket(b"")
        client._socket_response_count = 1

        with self.assertRaises(RFLinkStaleConnectionError):
            client._receive_http_response()

    def test_broken_pipe_on_reused_socket_is_classified_as_stale_keep_alive(self):
        client = RFLinkClient("127.0.0.1", 18083)
        client.sock = Mock()
        client.sock.sendall.side_effect = BrokenPipeError("peer closed")
        client._socket_response_count = 1

        with self.assertRaises(RFLinkStaleConnectionError):
            client._send_exchange_request()

    def test_opening_new_transport_preserves_injected_controller_state(self):
        sock = ReceivingSocket()
        client = RFLinkClient("127.0.0.1", 18083)
        client._controller_started = True

        with patch("hoverpilot.rflink.client.socket.socket", return_value=sock):
            client._open_socket(log=False)

        self.assertTrue(client._controller_started)
        self.assertIs(client.sock, sock)
        self.assertEqual(sock.address, ("127.0.0.1", 18083))

    def test_close_only_restores_controller_after_successful_injection(self):
        client = RFLinkClient("127.0.0.1", 18083)
        client._restore_original_controller = Mock()

        client.close()

        client._restore_original_controller.assert_not_called()

        client._controller_started = True
        client.close()

        client._restore_original_controller.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
