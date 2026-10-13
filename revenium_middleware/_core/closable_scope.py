"""A context-local flag that ends with the ``with`` block that set it."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


class _Mark:
    __slots__ = ("open",)

    def __init__(self):
        self.open = True


class ClosableScope:
    """A flag that is set inside ``entered()`` and nowhere else.

    A plain ``ContextVar`` flag stays set in every context copied inside the
    block (a task the enclosed call starts, such as an async LiteLLM success
    callback) for as long as that copy lives. The mark here is shared with
    those copies and closed when the block ends, so what a copy does after the
    block returned is outside the scope.
    """

    def __init__(self, name: str):
        self._mark: ContextVar[Optional[_Mark]] = ContextVar(name, default=None)

    @contextmanager
    def entered(self) -> Iterator[None]:
        mark = _Mark()
        token = self._mark.set(mark)
        try:
            yield
        finally:
            mark.open = False
            self._mark.reset(token)

    def is_active(self) -> bool:
        mark = self._mark.get()
        return mark is not None and mark.open
