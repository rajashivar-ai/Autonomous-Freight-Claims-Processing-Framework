# Synthetic freight-claims reproducibility demonstration

**All claims, evidence, carriers, contracts, and event timestamps are simulated. This is a newly created demonstration, not the implementation, dataset, or experimental record underlying the submitted manuscript.** No real operational data were available to build it. It cannot verify the manuscript's 0.890 real-data F1, 0.871 synthetic F1, 85% STP, economic claims, 40-trial searches, or independent re-annotation of 2,000 claims. Do not describe these files as historical records or independent validation.

## Run

Use Python 3.11 or newer in a virtual environment:

```sh
python -m pip install -r requirements.txt
python run_demo.py
```

The script regenerates the data, trains the models, selects configurations using validation data, evaluates once on held-out test data, and writes logs and predictions. Re-running overwrites generated files; copy any completed human annotation forms elsewhere first. Only NumPy is required. No credentials, SAP instance, external model services, network connection during execution, or GPU are required.

## Contents and provenance

- `data/synthetic_claims.csv`: 5,000 new simulated claims, 80 fictional carrier IDs, four regions, and unique freight orders. Every row carries a synthetic provenance flag.
- `data/simulated_events.jsonl`: the same cases packaged as timestamped event payloads. These are simulated arrivals across calendar year 2025, not captured live events or evidence of deployed throughput.
- `run_demo.py`, `config.json`, `requirements.txt`: executable generator, training/evaluation pipeline, fixed assumptions and search spaces, and dependency version.
- `models/*.npz`: weights and train-only standardization statistics from the actual demonstration runs. These are linear softmax models, not pretrained NLP or vision models.
- `logs/validation_trials.csv`: all 18 actual trials (six per model), including validation scores. No historical trials have been invented.
- `results/selected_configurations.json`: selected configurations. Ties choose the first configuration in declared search order.
- `results/test_predictions.csv`: actual predictions on 850 synthetic test cases for each model; class probabilities and confidence gate outcomes are included.
- `results/metrics.json`: full-test and subgroup metrics, ambiguity-aware accuracy, bootstrap intervals, and confidence coverage.
- `logs/run_log.json`: actual execution timestamps, environment, duration, and split counts. Execution timestamps are distinct from simulated event timestamps.
- `logs/validation_checks.json`: integrity checks executed by the pipeline.
- `annotations/blinded_synthetic_evidence.csv`: 150 randomly sampled synthetic test cases without generator labels or model predictions.
- `annotations/independent_annotation_form_BLANK.csv`: 450 empty rows, three reviewer slots per sampled case. These are forms, not completed annotations.
- `sha256_manifest.json`: file integrity hashes. Re-running changes runtime logs and the manifest.

## Generator assumptions

The class probabilities 0.412/0.386/0.202 are copied from aggregate proportions stated in the supplied manuscript solely as simulation inputs; exact sampled counts vary. All other distributional choices, noise levels, contractual values, and carrier/region choices are illustrative assumptions, not fitted to the provider's data. In particular, the multi-cause probability 0.18 is invented for simulation and is not an estimate from the manuscript. The source code gives every distribution and parameter.

The generator first samples a primary latent cause and optionally a second cause. It then samples noisy image-damage proxies, delay measurements, document quantities, and contract tolerances conditional on those causes. The primary cause is synthetic truth. The possible-label set is generator truth, not independent expert judgment. No raw photographs, OCR documents, SAP records, real contracts, or actual carrier histories are supplied. Evidence prose is templated from numeric measurements; models do not process that prose.

Because the simulation explicitly links contract-normalized signals to the latent label, any advantage for normalization partly reflects generator design. It cannot demonstrate that SAP master-data integration improves real-world performance. No real contractual dependence structure is reproduced.

## Models and evaluation specification

All three models are multinomial logistic regression implemented as a NumPy softmax model, initialized at zero, fitted with full-batch gradient descent, an unpenalized intercept, and no class weighting. L2 regularization applies to feature weights. Inputs are standardized using training statistics only. There is no stochastic optimizer or random model initialization.

| Model | Ordered feature columns before standardization |
|---|---|
| D1_raw_proxies | image_damage_proxy; delay_hours; shipped_quantity minus extracted_received_quantity; document_quality |
| D2_contract_normalized_proxies | image_damage_proxy; delay_hours / sla_buffer_hours; (shipped_quantity minus extracted_received_quantity) / shipped_quantity minus contractual_loss_tolerance; document_quality |
| D3_without_image_proxy | D2 features with image_damage_proxy removed |

D1–D3 are demonstration models, not B1–B5 or reproductions of any cited published method. D2 versus D3 removes one feature while keeping the family and tuning budget fixed; configurations are independently selected. This is not an exact replication of the paper's ablation. D1 versus D2 changes two feature constructions and cannot isolate a single change.

The deterministic chronological split is 3,350 training, 800 validation, and 850 test claims. Each unique freight order appears in only one split. Select from L2 values [0, 0.001, 0.01] and learning rates [0.03, 0.1], with 350 epochs per trial. Choose the highest validation macro F1 and preserve the train-fitted weights; do not retrain on validation data. No test values enter fitting or selection. Labels, possible-label sets, timestamps, and IDs are excluded from features.

Predictions use argmax over damage, delay, shortage, in that order. The confidence threshold is fixed at 0.8, not chosen to reproduce 85% automation. Its coverage measures classification confidence only: it is **not STP, liability accuracy, calibrated confidence, or financial posting**. There is no adjudication or enterprise write-back in this package.

Macro F1 averages the three class F1 values with zero for an undefined class F1. The reported 95% intervals are percentile intervals from 500 claim-level bootstrap resamples of one fitted model's test predictions. They do not account for training variation or carrier clustering. Subgroups cover region and synthetic ambiguity, with denominators; small groups should not be interpreted as external validation. Ambiguity-aware accuracy accepts predictions matching either generator cause and is not multi-label F1. No independently validated-subset metric is reported because none exists.

The predeclared same-environment repeatability tolerance is absolute macro-F1 difference <= 0.000001. A rerun of this code checks deterministic repeatability only. Recovery by an independent implementation or independent organization has not been established. Performance values are determined by execution, not selected to match the manuscript.

## Data dictionary

`provenance`: origin flag; `claim_id`/`freight_order_id`/`carrier_id`: fictional identifiers; `event_timestamp`: simulated ISO UTC time; `split`: chronological partition; `region`: fictional operating region; `claim_value_usd`: simulated positive amount; `contract_payment_days`: illustrative payment term; `document_quality`: synthetic score in [0,1]; `image_damage_proxy`: noisy scalar standing in for image evidence; `delay_hours`: nonnegative transit deviation; `sla_buffer_hours`: positive contractual buffer; `shipped_quantity`: positive piece count; `extracted_received_quantity`: simulated noisy document quantity, which may exceed shipped quantity; `contractual_loss_tolerance`: allowed loss fraction; `synthetic_primary_label`: generator primary cause; `synthetic_possible_labels`: pipe-delimited generator cause set; `synthetic_ambiguous`: whether two causes were generated; `evidence_text`: templated simulated description.

## Human annotation protocol

Provide only the blinded evidence packet and a separate form copy to each real reviewer. Keep generator truth and predictions inaccessible until labels are locked. Reviewer slots 1–3 are placeholders, not invented people. Reviewers should record an identifier, primary cause (damage/delay/shortage/uncertain), any additional causes, confidence (high/medium/low), evidence-based rationale, actual completion time, and status. Permit uncertain labels rather than forcing agreement. Record disagreements before reconciliation. A high-confidence subset definition should be fixed before comparing model performance, for example unanimous primary cause and high confidence from all three reviewers.

No independent reviewer has completed this exercise. Even genuine annotation of this packet would validate only synthetic evidence, not the private real corpus. To address the journal's real-data concern, apply a comparable blinded procedure to actual held-out test cases with appropriate access and retain the real records.

## Appropriate manuscript wording

“We additionally provide a newly developed, explicitly synthetic demonstration of the data-generation and evaluation workflow. Its data and simplified classifiers do not reproduce the proprietary deployment or substantiate the previously reported real-data results. Logs and predictions were produced by executing the supplied scripts. Independent replication and independent human annotation remain to be completed.”

Only claim public availability after a repository is actually released and verified. This package has not been uploaded to GitHub. Before journal submission, reconcile the manuscript's original implementation, historical metrics, supplementary-material claims, and new demonstration with the actual evidence available. The IJIES Word template governs manuscript formatting; it does not authenticate experimental records.
