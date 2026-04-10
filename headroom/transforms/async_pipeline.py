"""Async transform pipeline for parallel execution.

This module provides utilities for executing transforms in parallel
using asyncio.gather(). Transforms that support async execution can
run concurrently without blocking the event loop.

Design:
- Each transform can declare dependencies on other transforms
- Transforms without dependencies can run in parallel
- Use asyncio.gather() with return_exceptions=True for concurrent execution
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Transform
    from ..config import TransformResult
    from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def build_dependency_graph(transforms: list["Transform"]) -> dict[str, list[str]]:
    """Build a dependency graph from transforms.

    Returns:
        Dict mapping transform name -> list of dependency names.
    """
    graph = {}
    for transform in transforms:
        # Each transform declares its dependencies
        deps = getattr(transform, 'dependencies', lambda: [])()
        graph[transform.name] = deps if deps else []
    return graph


def get_execution_stages(transforms: list["Transform"]) -> list[list["Transform"]]:
    """Group transforms into stages that can run in parallel.

    Returns list of stages, where each stage is a list of transforms
    that can run in parallel (no dependencies between them).

    Example:
        [Stage1: [t1, t2], Stage2: [t3], Stage3: [t4, t5]]
        means: t1 and t2 run in parallel, then t3, then t4 and t5 in parallel
    """
    if not transforms:
        return []

    # Build name -> transform mapping
    name_to_transform = {t.name: t for t in transforms}

    # Build dependency graph
    graph = {}
    remaining = set()
    for t in transforms:
        deps = getattr(t, 'dependencies', lambda: [])()
        graph[t.name] = deps if deps else []
        remaining.add(t.name)

    # Group into stages
    stages: list[list[Transform]] = []

    while remaining:
        # Find transforms with no remaining dependencies
        ready = [
            name_to_transform[name]
            for name in remaining
            if all(dep not in remaining for dep in graph[name])
        ]

        if not ready:
            # Circular dependency or error - fall back to sequential
            logger.warning("Circular dependency detected, using sequential fallback")
            remaining_names = list(remaining)
            stages.append([name_to_transform[name] for name in remaining_names])
            break

        stages.append(ready)
        for t in ready:
            remaining.discard(t.name)

    return stages


async def run_transform_async(
    transform: "Transform",
    messages: list[dict[str, Any]],
    tokenizer: "Tokenizer",
    **kwargs: Any,
) -> "TransformResult":
    """Run a single transform asynchronously.

    Falls back to thread pool executor for CPU-bound transforms.
    """
    # Check if transform has async_apply method
    if hasattr(transform, 'async_apply'):
        result = transform.async_apply(messages, tokenizer, **kwargs)
        if result is not NotImplemented:
            return await result

    # Fall back to thread pool for CPU-bound work
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: transform.apply(messages, tokenizer, **kwargs)
    )


async def apply_transforms_parallel(
    transforms: list["Transform"],
    messages: list[dict[str, Any]],
    tokenizer: "Tokenizer",
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], list[str], dict[str, float]]:
    """Apply transforms in parallel where possible.

    Transforms are grouped into stages based on dependencies.
    Within each stage, transforms run in parallel via asyncio.gather().
    Stages execute sequentially.

    Args:
        transforms: List of transforms to apply.
        messages: Messages to transform.
        tokenizer: Tokenizer for token counting.
        **kwargs: Additional arguments passed to transforms.

    Returns:
        (transformed_messages, transforms_applied, timing)
    """
    if not transforms:
        return messages, [], {}

    # Get execution stages
    stages = get_execution_stages(transforms)
    logger.debug(f"Pipeline execution stages: {len(stages)} stages")

    current_messages = messages
    all_transforms: list[str] = []
    all_timing: dict[str, float] = {}

    for stage_idx, stage in enumerate(stages):
        stage_start = time.perf_counter()

        # Filter transforms that should apply
        applicable = [
            (t, t.should_apply(current_messages, tokenizer, **kwargs))
            for t in stage
        ]
        stage_transforms = [t for t, should in applicable if should]

        if not stage_transforms:
            continue

        # Run all transforms in this stage in parallel
        tasks = [
            run_transform_async(t, current_messages, tokenizer, **kwargs)
            for t in stage_transforms
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for t, result in zip(stage_transforms, results):
            if isinstance(result, Exception):
                logger.error(f"Transform {t.name} failed: {result}")
                # Continue with other transforms
                continue

            # Use the result messages for the next stage
            current_messages = result.messages
            all_transforms.extend(result.transforms_applied)

        stage_duration = (time.perf_counter() - stage_start) * 1000
        all_timing[f"stage_{stage_idx}"] = stage_duration
        logger.debug(
            f"Pipeline stage {stage_idx + 1}: {len(stage_transforms)} transforms "
            f"in {stage_duration:.1f}ms"
        )

    return current_messages, all_transforms, all_timing


def log_transform_timing(timing: dict[str, float]) -> None:
    """Log transform timing breakdown.

    Args:
        timing: Dict of transform_name -> duration_ms
    """
    if not timing:
        return

    parts = []
    for name, duration in sorted(timing.items(), key=lambda x: -x[1] if x[1] else 0):
        if duration and duration > 0.1:  # Only log significant timings
            parts.append(f"{name}={duration:.1f}ms")

    if parts:
        logger.info(f"Transform timing: {' '.join(parts)}")
