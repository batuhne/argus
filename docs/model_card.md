# Model card: Argus fraud classifier

A reference for what the champion model is, what it is for, how well it does, and
where it should not be trusted. Numbers below are from the current champion; rerun
`make train` and `make backtest` to refresh them.

## Model details

- Registry name `argus_fraud_classifier`, current champion version 1, family XGBoost.
- Gradient-boosted decision trees, chosen on validation AUPRC against LightGBM and
  CatBoost candidates over an Optuna sweep.
- Output is a calibrated fraud probability (isotonic calibration) plus a binary
  decision at a tuned threshold.
- Trained and tracked with MLflow. Every version carries its hyperparameters,
  per-split metrics, the encoder and calibrator artifacts, the feature schema, and
  lineage tags (git SHA, DVC lock hash, environment hash).

## Intended use

- Score card-not-present transactions in near real time and flag the riskiest for
  review, under a fixed alert budget.
- It is a decision-support signal, not an automated decline. A flag should route to
  a review step, not block a customer outright.
- Out of scope: any use outside the IEEE-CIS transaction shape, and any setting
  where the cost trade-off differs materially from the one below.

## Training data

- IEEE-CIS fraud detection dataset, about 590k transactions with a fraud rate near
  3.5 percent.
- Split chronologically into train, validation, test, and a final holdout window.
  Order of use is train (fit), validation (model and hyperparameter choice), test
  (threshold and promotion gate), holdout (untouched until the backtest).

## Features

117 features in four groups, all defined once in
`src/fraud/transforms/feature_logic.py`:

- 8 velocity and amount features assembled at serving: 5 card-velocity aggregates
  from the Feast Redis online store (point-in-time correct in the offline training
  join), 2 derived on demand at query time (amt_log, amt_to_card_mean_24h), and
  TransactionAmt taken straight from the request.
- 33 per-transaction numerics (curated C, D, dist, addr columns) passed to the
  trees with missing values left intact for native handling.
- A reduced 30-column V block, frozen once from the train split.
- 46 columns from 23 raw categoricals, encoded by a frequency map and an
  out-of-fold smoothed target mean fit on train only, so a row's encoding never
  sees its own label.

The encoder is persisted and loaded at serving, and a skew test asserts the offline
and online feature values match, so there is no train-serve skew.

## Evaluation

Test split (drives the promotion gate):

- AUPRC 0.54.
- At the tuned threshold (0.074): recall 0.67, precision 0.30, flagged rate 0.091.

Holdout window (untouched by training, threshold, and gate; 88,581 transactions,
3,083 fraudulent, base rate 0.035):

- AUPRC 0.45, well above the 0.035 base rate. The drop from the 0.54 test figure is
  expected: the holdout is later in time and never seen during tuning.
- Recall at alert budgets: 0.13 at 0.5 percent, 0.24 at 1 percent, 0.51 at 5 percent.
- Expected cost 1.74 USD per transaction under the cost matrix below.
- Calibration Brier score 0.024.

The Kaggle competition test set is unlabeled, so it cannot act as a fresh-data
evaluator. The chronological holdout is the honest proxy: it is genuinely unseen by
the model. It does sit in the same broad distribution as the rest of the data rather
than the competition's later, shifted test period, so holdout AUPRC is not directly
comparable to a Kaggle leaderboard score.

## Cost model and threshold

- Cost matrix: a missed fraud (false negative) costs 100 USD, a false alarm (false
  positive) costs 5 USD.
- The threshold is chosen on the test split to minimize expected cost under a recall
  floor and an alert-volume budget, not to maximize raw accuracy. Both the threshold
  and the gate operate on calibrated scores.

## Limitations

- Trained on one historical dataset; it has not seen real concept drift, only the
  chronological passage within that dataset and, for alert testing, injected shifts.
- Performance is reported in-distribution. A genuinely shifted production period
  would likely score lower until retraining catches up.
- The velocity features depend on Redis being current; stale or missing online
  features degrade scores for the affected cards. Cold-start cards fall back to
  neutral velocity values.
- Calibration holds near the observed score range; far-tail scores are less certain.

## Monitoring and maintenance

- A monitoring service tracks rolling AUPRC against later-arriving labels and
  population stability (PSI) on the model's input features.
- Sustained AUPRC below its floor publishes a drift alert, which a bridge turns into
  a retraining run through the same promotion gate. The same flow also runs on a
  schedule.
- Retraining never auto-promotes: a fresh candidate ships only if it beats the
  champion on the gate's AUPRC and cost criteria.

## Ethical considerations

- Fraud models can encode bias from historical labels. The feature set is
  transaction-centric and excludes direct demographic attributes, but proxy effects
  are still possible and a flag is meant to trigger review, not an automatic adverse
  action.
- Decisions and their inputs are logged for audit, so a contested flag can be traced
  back to the score, threshold, and features behind it.
