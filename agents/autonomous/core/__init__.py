# Autonomous Agent Core Modules

import concurrent.futures


def call_with_timeout(fn, args=(), kwargs=None, timeout=15):
    """Call *fn* with a per-request timeout (seconds).

    Raises TimeoutError if the call does not complete within *timeout*.
    Prevents OS-default socket timeouts (~127s) from eating the agent's
    entire time budget on a single failed request.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)
