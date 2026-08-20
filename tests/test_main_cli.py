import pytest

from hoverpilot.main import parse_args


def test_demo_defaults_to_a_bounded_run():
    args = parse_args([])

    assert args.steps == 100
    assert args.forever is False
    assert args.throttle == 0.55


def test_demo_can_explicitly_run_forever():
    args = parse_args(["--forever", "--throttle", "0.6"])

    assert args.steps == 100
    assert args.forever is True
    assert args.throttle == 0.6


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_demo_rejects_out_of_range_throttle(value):
    with pytest.raises(SystemExit):
        parse_args(["--throttle", value])
