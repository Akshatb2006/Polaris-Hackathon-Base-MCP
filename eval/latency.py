"""Latency breakdown over MCP HTTP: python eval/latency.py URL

Runs the organizer-style empty-query pool at concurrency 1 and 5 and reports each
workload separately (p50/p95/p99/max), so tail latency can be attributed.
"""
import asyncio
import math
import random
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import random_filter_args  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[math.ceil(q * len(xs)) - 1]


async def main(url):
    pool = [random_filter_args(random.Random(1000 + i)) | {'limit': 10, 'offset': 0} for i in range(50)]
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as s:
            await s.initialize()
            for a in pool[:20]:
                await s.call_tool('search_products', a)
            for conc in (1, 5):
                sem, times = asyncio.Semaphore(conc), []

                async def one(a):
                    async with sem:
                        t = time.perf_counter()
                        await s.call_tool('search_products', a)
                        times.append((time.perf_counter() - t) * 1000)
                for _ in range(3):
                    await asyncio.gather(*(one(a) for a in pool))
                print(f'concurrency {conc}: n={len(times)} p50 {pct(times, .5):.1f}  p95 {pct(times, .95):.1f}  '
                      f'p99 {pct(times, .99):.1f}  max {max(times):.1f} ms')


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1]))
