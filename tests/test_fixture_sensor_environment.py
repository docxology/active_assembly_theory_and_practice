import pytest

from src.fixture_sensor_environment import FixtureSensorEnvironment


def test_real_selected_acquisition_and_hidden_identity():
    for name, expected in [("x86_accumulator", 3), ("x86_bitwise", 2)]:
        env = FixtureSensorEnvironment(name)
        assert env.reset() == (0, 0)
        assert env.acquisition_log == []
        assert env.step(1) == (1, 1)
        assert env.acquisition_log == ["static"]
        assert env.step(2) == (expected, 2)
        assert env.acquisition_log == ["static", "dynamic"]
        assert env.step(0) == (0, 0)
        with pytest.raises(ValueError, match="SENSOR_INVALID_ACTION"):
            env.step(3)
    with pytest.raises(ValueError, match="SENSOR_UNSUPPORTED_FIXTURE"):
        FixtureSensorEnvironment("arbitrary")


def test_acquisition_log_bound():
    env = FixtureSensorEnvironment("x86_accumulator")
    for _ in range(100):
        env.step(1)
    with pytest.raises(ValueError, match="SENSOR_ACQUISITION_LIMIT"):
        env.step(1)
    assert len(env.acquisition_log) == 100
    assert env.step(0) == (0, 0)
