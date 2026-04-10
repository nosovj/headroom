"""SmartCrusher compression pipeline - decomposed.

Decomposed compression with clean layer interfaces:
- parse: JSON parsing, array type classification
- analyze: Statistical analysis, entropy, change point detection
- compress: Array and object crushing
- serialize: JSON output with caching
- pipeline: Compose layers into a pipeline
- gpu: GPU-accelerated implementations

This module contains the extracted and refactored functions from
headroom/transforms/smart_crusher.py for better testability and
GPU acceleration support.
"""

from headroom.compression.smart.analyze import (
    FieldStats,
    analyze_field,
    calculate_string_entropy,
    detect_change_points,
    detect_sequential_pattern,
    is_uuid_format,
)
from headroom.compression.smart.compress import (
    CompressConfig,
    CompressResult,
    crush_array,
    crush_string_array,
    crush_number_array,
    create_smart_crusher,
)
from headroom.compression.smart.parse import (
    ArrayType,
    classify_array,
    is_json_string,
    parse_json_content,
    safe_json_loads,
)
from headroom.compression.smart.pipeline import (
    CompressionPipeline,
    PipelineConfig,
    create_pipeline,
)
from headroom.compression.smart.serialize import (
    json_dumps,
    json_loads,
    orjson_available,
    serialize_compressed,
)

__all__ = [
    # parse
    "ArrayType",
    "classify_array",
    "is_json_string",
    "parse_json_content",
    "safe_json_loads",
    # analyze
    "FieldStats",
    "analyze_field",
    "calculate_string_entropy",
    "detect_change_points",
    "detect_sequential_pattern",
    "is_uuid_format",
    # compress
    "CompressConfig",
    "CompressResult",
    "crush_array",
    "crush_string_array",
    "crush_number_array",
    "create_smart_crusher",
    # pipeline
    "CompressionPipeline",
    "PipelineConfig",
    "create_pipeline",
    # serialize
    "json_dumps",
    "json_loads",
    "orjson_available",
    "serialize_compressed",
]
