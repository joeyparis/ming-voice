"""Keep all native model operations on one persistent Metal-owning thread."""
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Thread, current_thread
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
_owner: Thread | None = None


def _initialize() -> None:
    global _owner
    _owner = current_thread()


_worker = ThreadPoolExecutor(1, thread_name_prefix="ming-metal", initializer=_initialize)


def on_metal(function: Callable[P, R]) -> Callable[P, R]:
    """Serialize native operations; nested model calls stay on the same thread."""
    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        if current_thread() is _owner:
            return function(*args, **kwargs)
        return _worker.submit(function, *args, **kwargs).result()
    return run
