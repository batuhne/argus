# Argus

Real-time fraud detection built as a production-grade ML system, not a notebook model.

Argus covers the full lifecycle: streaming data ingestion, a feature store that
keeps training and serving in sync, reproducible training with experiment tracking
and a model registry, business-aware evaluation, low-latency serving, drift
monitoring, and a drift-triggered retraining loop wired into CI/CD. It runs
locally on Docker Compose and is designed to move to the cloud without a rewrite.

The dataset is the IEEE-CIS fraud detection set (about 590k transactions, heavy
class imbalance). The design targets Google MLOps maturity Level 2.

## Status

Runs end to end on Docker Compose: bring up the stack, train a champion, serve it,
replay transactions, and watch drift and rolling performance until a sustained drop
trips a retrain. It is a single-host system sized for a laptop, so a few things are
deliberately scaled down: the stream replays on a time warp rather than in real
time, drift can be exercised with injected shifts, and Redis runs as one node. Each
of those has a documented production path below.

## Architecture

```
Offline pipeline (DVC DAG)

  download -> validate -> clean -+-> split -+-> train / val / test -> train
                                 |          +-> holdout ------------> backtest
                                 |          +-> select V block -----> v_selected.json (-> train, backtest)
                                 |
                                 +-> build features -> Feast offline store -> materialize -> Redis (online)
                                          (card_features -> train, backtest)

  train: XGBoost / LightGBM / CatBoost, Optuna sweep, isotonic calibration
       -> promotion gate (AUPRC + cost) -> MLflow registry: champion
  backtest: score the champion on holdout, read-only -> report

Online loop (Redpanda)

  producer (replays holdout) -> transactions -> consumer -> POST /predict (BentoML)
                                                                |  velocity from Redis
                                                                v
                                                          predictions
                                                                |
  label-sim -> delayed labels --------------------> monitor: join, rolling AUPRC, PSI
                                                                |
                                                        drift alert -> retrain trigger -> train + gate (loop)
```

## Stack

Prefect, MLflow, Feast (Redis online store), DVC, Pandera, Redpanda, XGBoost,
LightGBM and CatBoost, BentoML, Evidently, Prometheus and Grafana, GitHub Actions,
and Kubernetes manifests rendered with Kustomize (validated locally on k3d).

## Getting started

Requires Docker, and `uv` for the Python toolchain.

```
cp .env.example .env   # fill in KAGGLE_API_TOKEN
uv sync
```

`uv sync` installs the full toolchain (serving runtime plus the training and data
tooling). The serving and monitoring image installs only the runtime dependencies,
so it stays small and clean of training-only packages.

## Data

The IEEE-CIS data comes from a Kaggle competition. Accept the rules once at
https://www.kaggle.com/competitions/ieee-fraud-detection/rules, then put your
Kaggle token in `.env` as `KAGGLE_API_TOKEN`.

Build the dataset by running the pipeline through the split stage, which
downloads, validates, cleans, and splits the data:

```
uv run dvc repro split
```

Splits are chronological by transaction time rather than random, so the model is
never trained on transactions that occur after the ones it is evaluated on. The
four splits are train, validation, test, and a final holdout window that nothing
in training, threshold selection, or the promotion gate is allowed to see. Raw and
processed data are tracked with DVC and stay out of git.

## Feature store

Every feature is defined once as a pure transform in
`src/fraud/transforms/feature_logic.py`. The offline feature table and the Feast
feature views both build on that single module, so a transaction is described the
same way during training and during serving. A skew test
(`tests/model/test_train_serve_skew.py`) asserts the offline and online values
agree to a tight tolerance, including unseen-category and missing-value cases.

The model reads 117 features in four classes, each on its correct path:

- Per-card velocity aggregates (counts, sums, time since last transaction) live in
  the Feast online store and are joined point-in-time correct during training.
- Per-transaction numerics (the curated C, D, dist, addr columns) are sent on the
  request and reach the trees with missing values left intact.
- A reduced V block, frozen once from the train split, rides the request the same way.
- Per-transaction categoricals (card type, email domains, the M flags, device and
  selected identity fields) are encoded by a fitted encoder. The encoder learns a
  frequency map and an out-of-fold smoothed target mean on the train split only,
  then is persisted as an MLflow artifact and loaded at serving exactly like the
  calibrator, so a row's encoding never leaks its own label.

The reduced V block is frozen once from the train split (drop near-empty columns,
cluster correlated ones, keep one representative each) and written to
`feature_repo/v_selected.json`, so the V columns the model uses are fixed and
versioned rather than chosen anew on every run.

`uv run dvc repro features` builds the offline feature table. To also load the
latest values per card into Redis for online lookups, bring up the stack and
materialize:

```
make up
make features
```

## Training

Training fits three gradient-boosted candidates, XGBoost, LightGBM and CatBoost,
over an Optuna AUPRC sweep, and picks the best on the validation split. Each run
logs hyperparameters, AUPRC and recall@k for every split, the PR curve, a SHAP
summary, the encoder and calibrator artifacts, the feature schema, and lineage tags
(git SHA, DVC lock hash, locked environment hash) so any registered model can be
traced back to the exact code and data behind it. The winner is registered as a new
version of the `argus_fraud_classifier` model and tagged `candidate`.

```
make up
make features
make train
```

A promotion gate compares the candidate against the current champion on the test
split (AUPRC and expected business cost) and promotes only on a genuine win.
Tunable knobs live under `training:` in `params.yaml`. The MLflow tracking server
is at `http://localhost:5500` while the stack is up.

## Evaluation

Beyond the test-split metrics that drive the gate, the champion is scored once more
on the untouched holdout window through the exact serving path (encoder transform
and velocity join), so the report reflects how the model behaves on data no part of
training ever touched:

```
make backtest
```

It reports AUPRC, recall at several alert budgets, expected cost per transaction,
and calibration (Brier), and logs them to MLflow. On the current holdout the
champion reaches an AUPRC around 0.45 against a fraud base rate near 0.035. The drop
from the test split is expected: the holdout is later in time and never seen during
training, threshold selection, or the gate.

The Kaggle competition test set is unlabeled, so it cannot serve as a fresh-data
evaluator. The chronological holdout is the honest proxy: it is genuinely unseen by
the model, even if it sits in the same distribution as the rest of the data rather
than the competition's later, shifted period.

`docs/model_card.md` has the full metrics, intended use, cost model, and limitations.

## Serving

The champion is served over REST with BentoML. A request carries the transaction
and its raw attribute vector; serving fetches the card's velocity features from
Redis, applies the same transforms and encoder as training, scores, calibrates, and
returns a probability and a decision at the tuned threshold. Readiness is gated on
the model and feature store being reachable.

```
make serve
```

## Streaming and high availability

The transaction stream runs on Redpanda. `make up` starts a three-broker cluster and
`make up-app` creates every topic at replication factor 3. Producers write with `acks=all`,
so a write needs a majority of its three replicas before it is acked, and one broker can
fail without losing or refusing writes. `redpanda-0` is the seed the other two bootstrap from, so it stays a cold-start single
point of failure on the demo, even though all three brokers are reachable from the host.
`make up-single` drops to one broker when a full cluster is too heavy, as on a laptop or CI.

Transactions are keyed by `card_id`, so a card's events stay on one partition and in order.
The consumer scales out: replicas in one group split the partitions. Redelivery is safe
because the monitor's score-to-label join is idempotent. The one exception to durability is
serving's inference log (the `scored-features` stream), which is best-effort and drops a
record rather than block a prediction, so its writes are outside the no-loss guarantee.

The monitor is a single replica because its join and PSI windows are in memory; scaling it
needs that state in a shared store first. Serving is stateless and scales out, as the
Kubernetes overlays show with a replica set and an autoscaler. Redis is single-node here;
production would front it with Sentinel or Cluster, a deployment change rather than a code one.

## Monitoring and retraining

A monitoring service joins predictions to the labels that arrive later, computes a
rolling AUPRC, and tracks population stability (PSI) on the model's input features.
It exposes Prometheus metrics; Grafana dashboards and Alertmanager rules ship with
the stack. To keep Prometheus cardinality bounded over more than a hundred features,
only the worst few PSI series are emitted per cycle alongside an aggregate.

```
make up-app
make produce        # replay transactions onto the stream
make label-sim      # replay the delayed ground-truth labels
make monitor        # rolling AUPRC and feature PSI
```

When rolling AUPRC stays below its floor for several cycles, the monitor publishes a
drift alert. A bridge turns sustained alerts into a retraining run (with a cooldown
so a burst collapses into one), which trains a fresh candidate and sends it through
the same promotion gate. The same flow runs on a schedule.

```
make retrain-serve     # hold the retraining deployment
make retrain-trigger   # bridge drift alerts to retraining
```

## Pipeline and reproducibility

The data and model pipeline is a DVC DAG: download, validate, clean, split,
select the V block, build features, train, and backtest. `uv run dvc repro` runs
only the stages whose inputs changed, and `dvc.lock` pins the exact data and code
hashes behind each output. Parameters live in `params.yaml`, so a sweep or a
threshold change is a tracked edit rather than a code change.

## Tests and CI

`make check` runs ruff, mypy in strict mode, and the test suite. The suite covers
the transforms and encoder, the feature contract, the train-serve skew, the
promotion gate and threshold logic, the streaming round-trips, the monitoring
join and drift, and the canary controller. GitHub Actions runs the same checks on
every push, gates model changes behind the skew and promotion tests against a Redis
service, and builds the serving image with a vulnerability scan that fails on
fixable HIGH or CRITICAL findings.
