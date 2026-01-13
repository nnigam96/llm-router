"""
Retry logic with exponential backoff for LLM client requests
"""

import asyncio
import time
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Last attempt failed, raise the exception
                        raise last_exception

                    # Calculate next delay with exponential backoff
                    time.sleep(min(delay, max_delay))
                    delay *= exponential_base

            # Should never reach here, but just in case
            raise last_exception

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Last attempt failed, raise the exception
                        raise last_exception

                    # Calculate next delay with exponential backoff
                    await asyncio.sleep(min(delay, max_delay))
                    delay *= exponential_base

            # Should never reach here, but just in case
            raise last_exception

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> Any:
    """
    Async function to retry another async function with exponential backoff.

    This is useful when you need retry logic without a decorator.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry on
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                # Last attempt failed, raise the exception
                raise last_exception

            # Calculate next delay with exponential backoff
            await asyncio.sleep(min(delay, max_delay))
            delay *= exponential_base

    # Should never reach here, but just in case
    raise last_exception
