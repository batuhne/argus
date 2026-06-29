# Runbooks

Operational responses for the alerts in `infra/prometheus/alerts.yml`. Each alert
carries a `runbook_url` pointing at the matching section here.

## Serving error budget burn

`ServingFastErrorBudgetBurn` (critical) and `ServingSlowErrorBudgetBurn` (warning)
fire when the 5xx ratio burns the 99.9% availability budget: fast is 14.4x over
1h and 5m, slow is 6x over 6h and 30m. Fast inhibits slow, so a paging burn does
not also raise the ticket.

- Check the SLO dashboard error-ratio and burn-rate panels to confirm the source.
- Read serving logs for the failing dependency: a champion-load failure, an MLflow
  outage, or a feature-fetch error (see the feature-fetch runbook).
- If a bad release is the cause, roll back `:stable` to the prior `sha-` image
  (see `infra/k8s/README.md`).
- If load is the cause, scale serving out; the HPA targets 70% CPU but caps at 6.

## Serving latency SLO breach

`ServingLatencySLOBreach` fires when p99 request duration exceeds the 50ms budget
for 5m.

- Split request latency from feature-fetch latency on the dashboard: a slow Redis
  shows up in `argus_feature_fetch_seconds` before the whole request.
- Check Redis health and the network path; a degraded Redis that still answers the
  readiness PING can blow the request-path budget.
- Confirm serving is not saturated (`bentoml_service_request_in_progress` near the
  `max_concurrency` cap means add replicas).

## Serving auth rejections

`ServingAuthRejections` fires when over 5% of requests return 401/403.

- A wrong or rotated API key is the usual cause; the caller is sending a stale
  bearer token. Confirm the consumer and serving share the same `SERVING_API_KEY`.
- Follow the API key rotation runbook to bring the two back in lockstep.

## Serving load shedding

`ServingLoadShedding` fires when over 1% of requests return 429 for 10m.

- The per-replica `max_concurrency` cap is shedding excess load. Add serving
  replicas (or raise the HPA ceiling) to absorb demand.
- Confirm the consumer is honoring `Retry-After` and backing off rather than
  amplifying the load.

## Serving feature fetch

`ServingFeatureFetchSlow` (p99 over 25ms) and `ServingFeatureFetchErrors` (over 1%
errors) isolate the Redis online-store read inside the request path.

- Check Redis latency and reachability; a slow but reachable Redis blows the
  request budget without failing the readiness PING.
- If Redis is down, serving readiness fails closed and pods leave the Service (see
  the Redis outage runbook).

## Feature drift detected

`FeatureDriftDetected` fires when any model-input feature PSI exceeds 0.2 versus
the training baseline.

- Open the drift dashboard; identify which feature shifted.
- A genuine population shift is expected to trigger a retrain through the
  `drift-alerts` topic; confirm a retraining run fired and gated.
- A single spiky feature with no performance impact can be a data-quality issue
  upstream; check the producer and source schema.

## Model performance decay

`ModelPerformanceDecay` fires when rolling AUPRC drops below the floor (0.30) with
enough matched labels to be meaningful.

- Confirm the labels are arriving (a label-pipeline stall starves the join and can
  look like decay; check the monitor runbook).
- A real drop is concept drift; a retrain should fire. Verify the gate decision
  (promote or keep-champion) in MLflow.

## Monitor exporter down or stalled

`MonitorExporterDown` fires when the monitoring target is unscrapable.
`MonitorJoinClockStalled` fires when the join clock trails wall-clock while events
are still flowing (gated on live ingest, so a finished replay reads as idle, not a
stall). `MonitorRecomputeStalled` fires when the recompute timestamp stops
advancing, catching a wedged loop that is still scrapable. `MonitorDriftComputeErrors`
fires when PSI computation is throwing on the drift worker, so feature-drift detection
is degraded even though fast metrics still flow.

- Down: check the monitoring container or pod; it rebuilds state from the retention
  window on restart, so a restart is safe. On k8s the monitor is single-replica with
  a Recreate strategy, so a routine deploy is briefly down; the alert waits 5m to ride
  that out, and a longer outage is the real signal.
- Behind or wedged: confirm the producer and label simulator are running, then check
  the consume loop for a hung poll. A restart rebuilds from the retention window.
- Drift failing: check the monitor logs for `drift_computation_failed`; a malformed
  baseline or an all-constant feature window is the usual cause. Fast metrics are unaffected.
- Schema mismatch: a spike in `argus_monitor_poison_messages_total` means messages are failing
  validation; check the producer schema against the event models. The loop runs but ingests nothing.

## Retrain dispatch failing

`RetrainDispatchFailing` fires when the drift-to-retraining bridge fails to dispatch
the Prefect deployment. The bridge logs the error and deliberately keeps the cooldown
open so the next alert retries, but until a dispatch succeeds, detected drift triggers
no retraining.

- Check the bridge logs for `retrain_dispatch_failed`; a Prefect API outage or a wrong
  `deployment_name` is the usual cause. Confirm the retraining deployment is served
  (`make retrain-serve`) and reachable.
- The bridge runs as host tooling, not in the slim runtime image (it needs the Prefect
  client from the pipeline dependency group). It exposes `argus_retrain_dispatch_failures_total`
  and `argus_retrain_dispatches_total` on its metrics port; point Prometheus at that port
  when running it as a long-lived service so this alert has data.
- Once the dependency is restored, the next drift alert dispatches and the counter stops
  advancing.

## API key rotation

Serving holds a single key, so dual-key rotation (accept the new key before
withdrawing the old) is not supported; a rotation always causes a brief auth
mismatch. Silence `ServingAuthRejections` for the rotation window, then rotate in
this order to keep the consumer authenticated:

1. Generate a new strong token.
2. Update the consumer's `SERVING_API_KEY` first; it holds messages on 401/403
   rather than dead-lettering, so a brief mismatch replays once serving catches up.
3. Update serving's Secret and roll the serving Deployment.
4. Confirm `ServingAuthRejections` clears, then remove the silence.

In k8s both consume `argus-serving-secrets`; update the Secret and roll both
workloads. In compose both read `SERVING_API_KEY` from the environment.

## Redis outage and serving readiness

Serving readiness PINGs Redis and fails closed: a fraud score without velocity
features is worse than no score, so a Redis outage removes serving from the Service
rather than serving degraded predictions.

- Expect serving pods to go NotReady after the probe failure threshold; the
  consumer holds messages and replays once Redis and serving recover.
- Restore Redis, then confirm pods return to Ready and the consumer drains its lag.

## Config and params rollout

`params.yaml` is parsed into a strict typed model (`src/fraud/params.py`): unknown
keys and out-of-range values are rejected at load, not silently ignored. Training,
serving, and monitoring all load it at startup, so a params change and the code
that reads it must ship in the same release.

- A mismatched pair fails fast at startup rather than running on stale behaviour: an
  old binary that meets a newly added key, or a new binary that meets a params file
  missing a now-required key, refuses to start.
- For serving, roll the change as one release. Readiness keeps the old replica in the
  Service until a new replica that accepts the new params is Ready, so a bad params
  change fails the new pods closed instead of draining the good ones.
- If a rollout wedges on a params error, check the startup logs for the rejected key
  or value, fix `params.yaml` to match the deployed code, and redeploy.
