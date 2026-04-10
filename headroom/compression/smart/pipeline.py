"""Compression pipeline composing all layers.

Pipeline: parse -> analyze -> compress -> serialize
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from headroom.compression.smart.analyze import analyze_field
from headroom.compression.smart.compress import (
    CompressConfig,
    CompressResult,
    crush_array,
    crush_string_array,
    crush_number_array,
)
from headroom.compression.smart.parse import ArrayType, classify_array, parse_json_content
from headroom.compression.smart.parse import ArrayType, parse_json_content

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for compression pipeline."""

    compress_config: CompressConfig | None = None
    min_tokens_to_compress: int = 200
    bias: float = 1.0


class CompressionPipeline:
    """Composable compression pipeline.

    Layers:
    1. Parse - JSON parsing and type detection
    2. Analyze - Statistical analysis
    3. Compress - Array/object crushing
    4. Serialize - JSON output
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or PipelineConfig()

    def compress_content(self, content: str) -> tuple[str, bool, str]:
        """Compress JSON content.

        Args:
            content: JSON string to compress.

        Returns:
            Tuple of (compressed_content, was_modified, strategy_info).
        """
        parsed, success, arr_type = parse_json_content(content)
        if not success:
            return content, False, "parse_failed"

        if arr_type is None:
            return content, False, "not_array"

        if arr_type == ArrayType.DICT_ARRAY:
            result = crush_array(parsed, self.config.compress_config)
            if result.was_modified:
                output = json.dumps(result.items)
                return output, True, result.strategy
            return content, False, result.strategy

        if arr_type == ArrayType.STRING_ARRAY:
            result = crush_string_array(parsed, self.config.compress_config)
            if result.was_modified:
                output = json.dumps(result.items)
                return output, True, result.strategy
            return content, False, result.strategy

        if arr_type == ArrayType.NUMBER_ARRAY:
            result = crush_number_array(parsed, self.config.compress_config)
            if result.was_modified:
                output = json.dumps(result.items)
                return output, True, result.strategy
            return content, False, result.strategy

        return content, False, f"type:{arr_type.value}"


def create_pipeline(config: PipelineConfig | None = None) -> CompressionPipeline:
    """Create a compression pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Configured CompressionPipeline.
    """
    return CompressionPipeline(config)
