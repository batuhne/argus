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

Early development. Not yet runnable end to end.

## Stack

Prefect, MLflow, Feast (Redis online store), DVC, Pandera, Redpanda, XGBoost and
LightGBM, BentoML, Evidently, Prometheus and Grafana, GitHub Actions, k3d, and
Terraform stubs.

## Getting started

Requires Docker, and `uv` for the Python toolchain.

```
cp .env.example .env   # fill in KAGGLE_API_TOKEN
uv sync
```

## Data

The IEEE-CIS data comes from a Kaggle competition. Accept the rules once at
https://www.kaggle.com/competitions/ieee-fraud-detection/rules, then put your
Kaggle token in `.env` as `KAGGLE_API_TOKEN`.

Build the dataset by running the pipeline, which downloads, validates, cleans,
and splits the data:

```
uv run dvc repro
```

Splits are chronological by transaction time rather than random, so the model is
never trained on transactions that occur after the ones it is evaluated on. Raw
and processed data are tracked with DVC and stay out of git.

## Feature store

Every feature is defined once as a pure transform in
`src/fraud/transforms/feature_logic.py`. The offline feature table and the Feast
feature views both build on that single module, so a transaction is described the
same way during training and during serving. A skew test
(`tests/model/test_train_serve_skew.py`) asserts the offline and online values
agree.

`dvc repro` builds the offline feature table. To load the latest values per card
into Redis for online lookups, bring up the stack and materialize:

```
make up
make features
```

Training data is assembled with a point-in-time correct join: for each
transaction, Feast joins the feature values as they stood at that moment, so the
join never reaches forward in time.

## Training

Training fits XGBoost as the primary model and LightGBM as a challenger over an
Optuna AUPRC sweep. Each run logs hyperparameters, AUPRC and recall@k for every
split, the PR curve, a SHAP summary, the feature schema, and lineage tags (git
SHA, DVC lock hash, locked environment hash) so any registered model can be
traced back to the exact code and data behind it. The best run is registered as
a new version of the `argus_fraud_classifier` model and tagged with the
`candidate` alias.

```
make up
make features
make train
```

Tunable knobs live under `training:` in `params.yaml` (Optuna trial budget and
timeout, SHAP sample size, recall@k levels, alias name). The MLflow tracking
server is at `http://localhost:5500` while the stack is up.

## Streaming and high availability

The transaction stream runs on Redpanda. `make up` starts a three-broker cluster and
`make up-app` creates every topic at replication factor 3. Producers write with `acks=all`,
so a write needs a majority of its three replicas before it is acked, and one broker can
fail without losing or refusing writes. `redpanda-0` is the seed and the only broker
published for host access, so it stays a cold-start single point of failure on the demo.
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

More to come (quickstart, demo script, screenshots).
