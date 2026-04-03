"""Progress bar and logging utilities."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


def get_progress() -> Progress:
    """Create a configured Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )


def track[T](
    iterable: Iterable[T],
    description: str = "Processing...",
    total: int | None = None,
    disable: bool = False,
) -> Iterator[T]:
    """Iterate with a progress bar."""
    if disable:
        yield from iterable
        return

    with get_progress() as progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)


@contextmanager
def status(message: str, disable: bool = False) -> Iterator[None]:
    """Show a status spinner while executing a block."""
    if disable:
        yield
        return

    from rich.console import Console

    console = Console()
    with console.status(message):
        yield
