"""Task functions executed by job workers.

Everything here must be importable *by the worker process* from its dotted path,
and must accept and return plain JSON-compatible types -- both sides of the broker
need to serialise them. Keep the bodies thin: they translate payloads and delegate
to a service.

Progress is reported through :func:`app.jobs.progress.report`, which resolves to
the in-process record or to the RQ job's metadata depending on who is running.
"""

from __future__ import annotations

from typing import Any

from app.jobs.progress import report
from app.schemas.pipeline import PipelineRequest
from app.services import pipeline as pipeline_service

#: Dotted path to :func:`run_pipeline`. Routers reference this rather than
#: retyping the string, so a rename cannot silently break enqueueing.
RUN_PIPELINE = "app.tasks.run_pipeline"


def run_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    """Run the full pipeline for a serialised :class:`PipelineRequest`."""
    parsed = PipelineRequest.model_validate(request)
    result = pipeline_service.run(parsed, report)
    return result.model_dump(mode="json")
