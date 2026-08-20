from unittest.mock import patch

from hoverpilot.doctor import main


def test_doctor_returns_success_when_endpoint_is_reachable():
    with patch("hoverpilot.doctor.check_tcp_connection") as check:
        result = main(["--host", "192.0.2.1", "--port", "18083"])

    assert result == 0
    check.assert_called_once_with("192.0.2.1", 18083, 2.0)


def test_doctor_returns_failure_when_endpoint_is_unreachable():
    with patch(
        "hoverpilot.doctor.check_tcp_connection",
        side_effect=TimeoutError("timed out"),
    ):
        result = main([])

    assert result == 1
