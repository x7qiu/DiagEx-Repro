import signal
import time

import pytest

from eval.pid2graph.deadline import wall_clock_limit


def test_stalled_operation_is_interrupted_and_handler_restored():
    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="wall-clock"):
        with wall_clock_limit(0.05):
            time.sleep(2)
    assert time.monotonic() - started < 1
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_existing_deadline_is_not_overwritten():
    with wall_clock_limit(10):
        with pytest.raises(RuntimeError, match="active outer deadline"):
            with wall_clock_limit(20):
                pass
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 10
