# Layer5_Streamlit_App/app.py

"""Interim Streamlit UI.

Phase 9 replaces this with React + TypeScript against the FastAPI service. Until
then it drives the same pipeline service the API and CLI use, so the three cannot
disagree about results.

Streamlit re-runs this whole script on every widget interaction, which is why the
run is behind an explicit button and cached by input. That limitation is
structural and is the reason the front end is being replaced.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from app.schemas.pipeline import PipelineRequest
from app.services import pipeline as pipeline_service
from app.services.forecasting import available_backends
from app.settings import get_settings
from utils.helpers import ensure_dirs

settings = get_settings()

st.set_page_config(layout="wide", page_title="Frontier")
st.title("Frontier")
ensure_dirs()

st.sidebar.header("Controls")
selected = st.sidebar.multiselect(
    "Universe", options=settings.universe, default=settings.universe[:6]
)
backend = st.sidebar.selectbox(
    "Forecast backend",
    options=available_backends(),
    help="'naive' is the random-walk baseline. 'keras_lstm' trains one model per "
    "ticker and takes minutes per ticker on CPU.",
)
start = st.sidebar.date_input("Start", dt.date.fromisoformat(settings.start_date))
end = st.sidebar.date_input("End", dt.date.fromisoformat(settings.end_date))
risk_tolerance = st.sidebar.slider("Risk tolerance", 0.0, 1.0, 0.5)
run = st.sidebar.button("Run pipeline", type="primary")


@st.cache_data(show_spinner=False)
def run_pipeline(tickers: tuple[str, ...], backend: str, start: dt.date, end: dt.date) -> dict:
    """Cached by inputs, so moving the risk slider does not retrain anything."""
    request = PipelineRequest(
        tickers=list(tickers), backend=backend, start=start, end=end
    )
    return pipeline_service.run(request).model_dump(mode="json")


def frame_to_pandas(frame: dict) -> pd.DataFrame:
    return pd.DataFrame(frame["data"], index=frame["index"], columns=frame["columns"])


if not run:
    st.info("Choose a universe and click **Run pipeline**.")
    st.stop()

if len(selected) < 2:
    st.error("Select at least two tickers.")
    st.stop()

with st.spinner(f"Running the {backend} pipeline..."):
    result = run_pipeline(tuple(selected), backend, start, end)

if result["failed"]:
    st.warning(
        "Skipped: "
        + ", ".join(f"{f['ticker']} ({f['reason']})" for f in result["failed"])
    )
for warning in result["warnings"]:
    st.warning(warning)

st.subheader("Forecast quality")
st.caption(
    "MASE below 1.0 beats a random walk; at or above 1.0 it does not. "
    "Directional accuracy of 0.5 is a coin flip. The legacy accuracy column is "
    "retained for continuity only -- a do-nothing forecast scores above 97 on it."
)
metrics = pd.DataFrame(result["forecast_metrics"]).set_index("ticker")
st.dataframe(
    metrics[
        [
            "mase_vs_naive",
            "directional_accuracy",
            "rmse",
            "mae",
            "r2",
            "legacy_approximate_accuracy",
        ]
    ],
    width="stretch",
)

st.subheader("Strategy performance")
performance = pd.DataFrame(result["performance"]).set_index("strategy")
st.dataframe(performance, width="stretch")

st.subheader("Weights by strategy")
st.dataframe(frame_to_pandas(result["weights"]), width="stretch")

st.subheader("Cumulative growth of 1.0 invested")
st.line_chart(frame_to_pandas(result["cumulative_growth"]))

# The slider is deliberately outside the cache key so that moving it never
# retrains anything -- which means the selection must be recomputed here rather
# than read from the cached result, whose value reflects the default tolerance.
ranked = performance["annual_volatility"].sort_values().index.tolist()
chosen_strategy = ranked[int(round(risk_tolerance * (len(ranked) - 1)))]

st.success(f"Risk tolerance {risk_tolerance:.2f} selects **{chosen_strategy}**")

weights = frame_to_pandas(result["weights"])[chosen_strategy]
weights = (weights / weights.sum()).sort_values(ascending=False)
st.bar_chart(weights.rename("weight"))
