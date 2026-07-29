import asyncio
from collections.abc import Coroutine
from typing import Any


async def gather_bounded(
    limit: int,
    *coros: Coroutine[Any, Any, Any],
    return_exceptions: bool = False,
) -> list[Any]:
    """`asyncio.gather` with a cap on how many coroutines run at once.

    Results come back in argument order, as with `asyncio.gather`. The cap keeps
    a burst of work (a game with hundreds of achievement badges, say) from
    opening a connection per item and starving everything else sharing the HTTP
    pool.

    Unlike `asyncio.gather`, a propagating error does not return while the peers
    of the failed coroutine are still running.
    """
    if limit < 1:
        # Closed rather than dropped, so rejecting the limit does not leave a
        # pile of un-awaited coroutines to warn about at collection time.
        for coro in coros:
            coro.close()
        raise ValueError("limit must be at least 1")

    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Coroutine[Any, Any, Any]) -> Any:
        async with semaphore:
            return await coro

    tasks = [asyncio.ensure_future(run(coro)) for coro in coros]

    try:
        return list(await asyncio.gather(*tasks, return_exceptions=return_exceptions))
    except Exception:
        # `asyncio.gather` completes as soon as one coroutine raises, leaving its
        # peers running and unobserved. Let them land rather than cancelling them:
        # a half-written file still satisfies the `*_exists` checks a later scan
        # skips on. Cancellation needs no such handling, since gather does cancel
        # its children and settles only once they have unwound.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
