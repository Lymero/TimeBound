"""Simple logging utility that provides basic logging functions with colorized output."""

from typing import Any

from rich.console import Console
from rich.traceback import install

install(show_locals=True)

console = Console()


def debug(message: str, **kwargs: dict[str, Any]) -> None:
    """Log a debug message.

    Args:
        message: The message to log
        **kwargs: Additional parameters to pass to rich.console.print
    """
    console.print(f"[bold magenta]DEBUG:[/] {message}", **kwargs)


def info(message: str, **kwargs: dict[str, Any]) -> None:
    """Log an informational message.

    Args:
        message: The message to log
        **kwargs: Additional parameters to pass to rich.console.print
    """
    console.print(f"[bold cyan]INFO:[/] {message}", **kwargs)


def success(message: str, **kwargs: dict[str, Any]) -> None:
    """Log a success message.

    Args:
        message: The message to log
        **kwargs: Additional parameters to pass to rich.console.print
    """
    console.print(f"[bold green]SUCCESS:[/] {message}", **kwargs)


def warning(message: str, **kwargs: dict[str, Any]) -> None:
    """Log a warning message.

    Args:
        message: The message to log
        **kwargs: Additional parameters to pass to rich.console.print
    """
    console.print(f"[bold yellow]WARNING:[/] {message}", **kwargs)


def error(message: str, **kwargs: dict[str, Any]) -> None:
    """Log an error message.

    Args:
        message: The message to log
        **kwargs: Additional parameters to pass to rich.console.print
    """
    console.print(f"[bold red]ERROR:[/] {message}", **kwargs)