"""The union of result payloads a job can carry.

``JobStatus.result`` is typed ``Any`` because the job store is generic: it runs
whatever task it is handed and keeps whatever that task returns. That is right
for the store, but it leaves the *API* under-described -- a client polling a run
has no published schema for what comes back, and the front end would have to
hand-maintain a copy of three response models.

This module names the three shapes a job actually produces and gives the result
endpoint a response model that lists them, so they appear in the OpenAPI schema
and generate into the front end's types. The union is also a check: a task whose
result matches none of these fails loudly here instead of reaching a browser as
an untyped blob.
"""

from __future__ import annotations

from frontier.api.schemas.backtest import BacktestResponse
from frontier.api.schemas.data import IngestResponse
from frontier.api.schemas.jobs import JobStatus
from frontier.api.schemas.pipeline import PipelineResult

#: Ordered widest-to-narrowest for readability only; pydantic matches on shape,
#: and the three have disjoint required fields.
JobResultPayload = PipelineResult | BacktestResponse | IngestResponse


class JobResult(JobStatus):
    """A completed job, with its result narrowed to the shapes tasks return."""

    result: JobResultPayload | None = None
