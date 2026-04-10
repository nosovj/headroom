"""Standalone compression server.

A minimal server that demonstrates the compression pipeline with:
- GPU memory management
- Model warmup at startup
- Health endpoint
- Concurrent request handling

Run on port 8788:
    python3 -m benchmarks.compression_server --port 8788
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Compression Service")


@dataclass
class CompressionStats:
    """Statistics for compression operations."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0

    def record(self, success: bool, latency_ms: float):
        self.total_requests += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1
        self.total_latency_ms += latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


stats = CompressionStats()
warmupper = None


class CompressionService:
    """Compression service with warmup."""

    def __init__(self):
        self.ready = False
        self._pipeline = None

    async def warmup(self, timeout: float = 60.0):
        """Warmup the compression service."""
        global warmupper
        from headroom.compression.smart import CompressionPipeline, PipelineConfig
        from headroom.compression.smart.warmup import CompressionWarmupper

        logger.info("Starting compression service warmup...")

        warmupper = CompressionWarmupper()
        await warmupper.warmup()

        self._pipeline = CompressionPipeline(PipelineConfig())
        self.ready = True
        logger.info("Compression service warmup complete")

    def compress(self, content: str) -> tuple[str, bool, str]:
        """Compress content.

        Returns:
            Tuple of (result, was_modified, strategy)
        """
        if not self.ready:
            raise RuntimeError("Service not ready - warmup incomplete")

        start = time.perf_counter()
        try:
            result, was_modified, strategy = self._pipeline.compress_content(content)
            latency = (time.perf_counter() - start) * 1000
            stats.record(True, latency)
            return result, was_modified, strategy
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            stats.record(False, latency)
            raise


service = CompressionService()


@app.on_event("startup")
async def startup():
    """Initialize compression service."""
    await service.warmup()


@app.get("/health/compression")
async def health_compression():
    """Health check for compression service."""
    if not service.ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "ready": False,
                "warmup_complete": warmupper.health_check() if warmupper else False,
            },
        )

    return {
        "status": "ready",
        "ready": True,
        "warmup_complete": True,
    }


@app.get("/health")
async def health():
    """Basic health check."""
    return {
        "status": "healthy" if service.ready else "starting",
        "compression": "ready" if service.ready else "loading",
    }


@app.get("/stats")
async def get_stats():
    """Get compression statistics."""
    return {
        "total_requests": stats.total_requests,
        "successful": stats.successful,
        "failed": stats.failed,
        "avg_latency_ms": round(stats.avg_latency_ms, 2),
    }


@app.post("/compress")
async def compress(request: dict):
    """Compress JSON content.

    Request body:
        {"content": "<json string>"}

    Response:
        {
            "compressed": "<compressed json>",
            "was_modified": true/false,
            "strategy": "strategy_name",
            "latency_ms": 123.45
        }
    """
    content = request.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    try:
        result, was_modified, strategy = service.compress(content)
        return {
            "compressed": result,
            "was_modified": was_modified,
            "strategy": strategy,
        }
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(port: int = 8788, host: str = "0.0.0.0"):
    """Run the compression server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Compression Server")
    parser.add_argument("--port", type=int, default=8788, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    logger.info(f"Starting compression server on {args.host}:{args.port}")
    run_server(port=args.port, host=args.host)
