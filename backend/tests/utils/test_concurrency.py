import asyncio
import inspect

import pytest

from utils.concurrency import gather_bounded


class _Tracker:
    """Records how many wrapped coroutines were running at the same time."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0
        self.completed: list = []

    async def run(self, value, ticks: int = 1):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        for _ in range(ticks):
            await asyncio.sleep(0)
        self.in_flight -= 1
        self.completed.append(value)
        return value


class TestGatherBounded:
    async def test_returns_results_in_argument_order(self):
        async def value(result, delay):
            await asyncio.sleep(delay)
            return result

        results = await gather_bounded(
            4, value("a", 0.03), value("b", 0.01), value("c", 0)
        )

        assert results == ["a", "b", "c"]

    @pytest.mark.parametrize(
        ("limit", "work", "expected_peak"),
        [(1, 4, 1), (3, 6, 3), (10, 3, 3)],
    )
    async def test_runs_up_to_the_limit_at_once(self, limit, work, expected_peak):
        tracker = _Tracker()

        await gather_bounded(limit, *(tracker.run(idx) for idx in range(work)))

        assert tracker.peak == expected_peak
        assert tracker.in_flight == 0

    async def test_propagates_the_first_exception(self):
        async def boom():
            raise ValueError("nope")

        async def fine():
            return "ok"

        with pytest.raises(ValueError, match="nope"):
            await gather_bounded(2, boom(), fine())

    async def test_a_failure_does_not_stop_its_peers(self):
        """One bad download must not cost a ROM the rest of its media."""
        tracker = _Tracker()

        async def boom():
            raise ValueError("nope")

        results = await gather_bounded(
            2,
            boom(),
            tracker.run("cover"),
            tracker.run("manual"),
            return_exceptions=True,
        )

        assert isinstance(results[0], ValueError)
        assert results[1:] == ["cover", "manual"]

    async def test_a_propagating_error_leaves_nothing_running(self):
        """A detached download would keep writing after the caller moved on."""
        tracker = _Tracker()

        async def boom():
            await asyncio.sleep(0)
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            await gather_bounded(
                3,
                boom(),
                tracker.run("cover", ticks=3),
                tracker.run("manual", ticks=3),
            )

        assert tracker.in_flight == 0
        assert sorted(tracker.completed) == ["cover", "manual"]

    async def test_no_coroutines_is_not_an_error(self):
        assert await gather_bounded(4) == []

    @pytest.mark.parametrize("limit", [0, -1])
    async def test_rejects_a_limit_below_one(self, limit):
        async def work():
            return "ok"

        rejected = work()

        with pytest.raises(ValueError, match="at least 1"):
            await gather_bounded(limit, rejected)

        assert inspect.getcoroutinestate(rejected) == inspect.CORO_CLOSED
