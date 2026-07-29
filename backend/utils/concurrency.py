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
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Coroutine[Any, Any, Any]) -> Any:
        async with semaphore:
            return await coro

    return list(
        await asyncio.gather(
            *(run(coro) for coro in coros), return_exceptions=return_exceptions
        )
    )
