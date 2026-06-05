"""Serve the retraining flow as a scheduled, drift-triggerable deployment."""

from __future__ import annotations

from fraud.params import load_params
from pipelines.flows.retraining_pipeline import RETRAIN_REASON_SCHEDULED, retraining_flow


def serve_retraining() -> None:
    cfg = load_params().retraining
    retraining_flow.serve(
        name="argus-retraining",
        cron=cfg.cron,
        parameters={"reason": RETRAIN_REASON_SCHEDULED},
    )


if __name__ == "__main__":
    serve_retraining()
