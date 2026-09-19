"""Experiment tracking contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrackedRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    run_name: str | None = None
    status: str
    started_at: str | None = None
    ended_at: str | None = None
    kind: str | None = Field(default=None, description="'pipeline' or 'backtest'.")
    git_commit: str | None = Field(
        default=None, description="Code that produced the run, when in a repository."
    )
    data_vintage: str | None = Field(
        default=None,
        description="When the underlying prices were ingested. Two runs over the same "
        "window can disagree because the provider restated history between them.",
    )
    params: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class TrackingStatus(BaseModel):
    enabled: bool
    tracking_uri: str | None = None
    experiment: str | None = None
    #: Present only when the store is reachable.
    run_count: int | None = None
    ui_command: str | None = Field(
        default=None, description="How to browse these runs locally."
    )
    detail: str | None = Field(
        default=None, description="Why tracking is unavailable, when it is."
    )


class TrackedRunList(BaseModel):
    runs: list[TrackedRun]
    total: int
