import asyncio

import pytest

from utils.concurrency import gather_bounded


class _Tracker:
    """Records how many wrapped coroutines were running at the same time."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def run(self, value, ticks: int = 1):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        for _ in range(ticks):
            await asyncio.sleep(0)
        self.in_flight -= 1
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

    async def test_runs_up_to_the_limit_at_once(self):
        tracker = _Tracker()

        await gather_bounded(3, *(tracker.run(idx) for idx in range(6)))

        assert tracker.peak == 3
        assert tracker.in_flight == 0

    async def test_never_exceeds_the_limit(self):
        tracker = _Tracker()

        await gather_bounded(1, *(tracker.run(idx) for idx in range(4)))

        assert tracker.peak == 1

    async def test_a_limit_above_the_work_runs_everything_at_once(self):
        tracker = _Tracker()

        await gather_bounded(10, *(tracker.run(idx) for idx in range(3)))

        assert tracker.peak == 3

    async def test_propagates_the_first_exception(self):
        async def boom():
            raise ValueError("nope")

        async def fine():
            return "ok"

        with pytest.raises(ValueError, match="nope"):
            await gather_bounded(2, boom(), fine())

    async def test_returns_exceptions_when_asked(self):
        async def boom():
            raise ValueError("nope")

        async def fine():
            return "ok"

        results = await gather_bounded(2, boom(), fine(), return_exceptions=True)

        assert isinstance(results[0], ValueError)
        assert results[1] == "ok"

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

        assert results[1:] == ["cover", "manual"]

    async def test_no_coroutines_is_not_an_error(self):
        assert await gather_bounded(4) == []

    @pytest.mark.parametrize("limit", [0, -1])
    async def test_rejects_a_limit_below_one(self, limit):
        with pytest.raises(ValueError, match="at least 1"):
            await gather_bounded(limit)
