from __future__ import annotations

from dataclasses import dataclass
import functools
import random
import time
from typing import Any, Callable, ParamSpec, TypeVar

from sva_toolkit.runtime.diagnostics import DEFAULT_DIAGNOSTICS, LOGGER

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = ()

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.backoff_base < 0:
            raise ValueError("backoff_base must be >= 0")
        if self.backoff_cap < 0:
            raise ValueError("backoff_cap must be >= 0")


class TransientResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code


class RetryExhaustedError(RuntimeError):
    def __init__(self, operation: str, retries: int, last_error: BaseException) -> None:
        self.operation = operation
        self.retries = retries
        self.last_error = last_error
        super().__init__(f"{operation} exhausted {retries} retr{'y' if retries == 1 else 'ies'}: {last_error}")


def retry(
    policy: RetryPolicy | Callable[..., RetryPolicy],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            resolved_policy = policy(*args, **kwargs) if callable(policy) else policy
            retries_used = 0

            while True:
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    retry_error = _coerce_retryable_error(exc, resolved_policy.retry_on)
                    if retry_error is None:
                        _set_retry_count(args, retries_used)
                        raise
                    if retries_used >= resolved_policy.max_retries:
                        _set_retry_count(args, retries_used)
                        diagnostics = _resolve_diagnostics(args)
                        diagnostics.record("retry_exhausted", detail=str(retry_error))
                        LOGGER.warning(
                            "%s exhausted %d retr%s: %s",
                            func.__qualname__,
                            retries_used,
                            "y" if retries_used == 1 else "ies",
                            retry_error,
                        )
                        raise RetryExhaustedError(func.__qualname__, retries_used, retry_error) from exc

                    retries_used += 1
                    delay = _compute_delay(resolved_policy, retries_used, retry_error.retry_after)
                    LOGGER.warning(
                        "retrying %s (%d/%d) after %.2fs: %s",
                        func.__qualname__,
                        retries_used,
                        resolved_policy.max_retries,
                        delay,
                        retry_error,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue

                _set_retry_count(args, retries_used)
                if retries_used > 0:
                    LOGGER.warning(
                        "%s succeeded after %d retr%s",
                        func.__qualname__,
                        retries_used,
                        "y" if retries_used == 1 else "ies",
                    )
                return result

        return wrapper

    return decorator


def _coerce_retryable_error(
    exc: Exception,
    retry_on: tuple[type[BaseException], ...],
) -> TransientResponseError | None:
    if isinstance(exc, TransientResponseError):
        return exc
    if isinstance(exc, retry_on):
        retry_after = getattr(exc, "retry_after", None)
        status_code = getattr(exc, "status_code", None)
        return TransientResponseError(str(exc), retry_after=retry_after, status_code=status_code)
    return None


def _compute_delay(policy: RetryPolicy, retry_number: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return max(0.0, min(policy.backoff_cap, retry_after))

    delay = min(policy.backoff_cap, policy.backoff_base * (2 ** max(0, retry_number - 1)))
    if policy.jitter:
        return random.random() * delay
    return delay


def _resolve_diagnostics(args: tuple[Any, ...]) -> Any:
    if args:
        diagnostics = getattr(args[0], "diagnostics", None)
        if diagnostics is not None and hasattr(diagnostics, "record"):
            return diagnostics
    return DEFAULT_DIAGNOSTICS


def _set_retry_count(args: tuple[Any, ...], retries_used: int) -> None:
    if args and hasattr(args[0], "last_retry_count"):
        setattr(args[0], "last_retry_count", retries_used)


__all__ = [
    "RetryExhaustedError",
    "RetryPolicy",
    "TransientResponseError",
    "retry",
]
