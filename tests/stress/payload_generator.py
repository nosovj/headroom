"""Payload generator for compression stress tests.

Generates realistic SmartCrusher inputs with configurable properties.
"""

import random
from typing import NamedTuple


class StressPayload(NamedTuple):
    """Generated payload for stress testing."""
    messages: list[dict]
    items: list[str]
    config: dict


class PayloadGenerator:
    """Generates realistic compression payloads.

    Configurable properties:
    - item_count: number of items (50-1000)
    - avg_chars: average characters per item (200-2000)
    - redundancy: fraction of near-duplicate items (0.0-0.8)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_items(
        self,
        item_count: int,
        avg_chars: int = 500,
        redundancy: float = 0.3,
    ) -> list[str]:
        """Generate text items with configurable redundancy.

        Args:
            item_count: Number of items to generate.
            avg_chars: Target average characters per item.
            redundancy: Fraction of items that are near-duplicates (0.0-0.8).

        Returns:
            List of text strings.
        """
        words_pool = [
            "function", "class", "method", "variable", "constant", "import",
            "data", "model", "input", "output", "result", "error", "handle",
            "process", "request", "response", "config", "settings", "options",
            "test", "validate", "check", "verify", "compute", "calculate",
            "update", "refresh", "load", "save", "read", "write",
            "debug", "log", "trace", "monitor", "measure", "profile",
        ]

        templates = [
            "{} processing completed with {} results",
            "Failed to {} due to {} error in {} handler",
            "Successfully {} {} using {} approach",
            "Warning: {} operation encountered {} issues",
            "Info: {} updated {} with {} configuration",
            "Debug: {} called {} method with {} parameters",
            "Result: {} validation passed for {} items",
            "Error: {} validation failed - missing {} field",
            "Completed {} operation in {}ms for {} records",
            "Starting {} process with {} configured options",
        ]

        items = []
        unique_count = int(item_count * (1 - redundancy))

        # Generate unique items
        for i in range(unique_count):
            length = int(avg_chars * self.rng.uniform(0.5, 1.5))
            words_in_item = max(3, length // 6)
            content = " ".join(
                self.rng.choices(words_pool, k=words_in_item)
            )
            template = self.rng.choice(templates)
            item = template.format(
                self.rng.choice(words_pool),
                self.rng.choice(words_pool),
                self.rng.choice(words_pool),
            )
            items.append(item + f" [item_{i}]")

        # Add near-duplicates
        for i in range(item_count - unique_count):
            base = self.rng.choice(items[:unique_count])
            # Create near-duplicate by changing a few words
            words = base.split()
            if len(words) > 3:
                for _ in range(self.rng.randint(1, 2)):
                    idx = self.rng.randint(0, len(words) - 1)
                    words[idx] = self.rng.choice(words_pool)
            items.append(" ".join(words) + f" [dup_{i}]")

        self.rng.shuffle(items)
        return items

    def generate_tool_output(
        self,
        n_items: int = 100,
        avg_chars: int = 500,
        redundancy: float = 0.3,
    ) -> list[dict]:
        """Generate tool output as a list of message items.

        Args:
            n_items: Number of items.
            avg_chars: Average characters per item.
            redundancy: Fraction of near-duplicates.

        Returns:
            List of message dicts with role and content.
        """
        items = self.generate_items(n_items, avg_chars, redundancy)
        return [
            {"role": "tool", "content": item, "tool": f"tool_{i % 5}"}
            for i, item in enumerate(items)
        ]

    def generate_mixed_payload(
        self,
        n_items: int = 100,
        avg_chars: int = 500,
        redundancy: float = 0.3,
    ) -> StressPayload:
        """Generate a full SmartCrusher input payload.

        Returns:
            StressPayload with messages, items, and config.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Process the following tool outputs efficiently."},
            *self.generate_tool_output(n_items, avg_chars, redundancy),
        ]

        items = self.generate_items(n_items, avg_chars, redundancy)
        config = {
            "item_count": n_items,
            "avg_chars": avg_chars,
            "redundancy": redundancy,
        }

        return StressPayload(messages=messages, items=items, config=config)