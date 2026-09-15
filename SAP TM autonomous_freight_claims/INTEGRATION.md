# Deployment and integration contracts

## 1. Durable workflow

Use separate states: RECEIVED -> EVIDENCE_PENDING -> READY -> APPROVED or REVIEW -> POST_PENDING -> POSTED -> SETTLED. Posting failure must not look like successful settlement. Add APPEAL_PENDING and reversal states without overwriting the original decision.

Suggested custom persistence (design specification, not delivered DDIC objects):

| Table | Key and essential fields |
|---|---|
| ZAFC_EVENT | MANDT + source_system + event_id unique; payload hash, claim ID, received UTC |
| ZAFC_CASE | MANDT + claim UUID; TOR key, version, state, contract version, decision ID |
| ZAFC_EVID | claim UUID + document UUID; SHA-256, repository ID, MIME type, collected UTC |
| ZAFC_DECISION | claim UUID + decision version; immutable input/output JSON, actor, UTC, policy/model hashes |
| ZAFC_OUTBOX | command UUID; claim/version, action, payload hash, attempts, next attempt, state |
| ZAFC_FI_REQ | ERP-side unique posting token; payload hash, status, company/year/document, update result |
| ZAFC_APPEAL | appeal UUID; claim/version, reason, independent reviewer, deadline and judgment |

Define client-dependent unique indexes and generated enqueue objects. Do not rely on a SELECT-before-INSERT check for uniqueness. Protect tables from ordinary direct maintenance; application-level append-only behavior alone is not immutable storage. Export decision/evidence hashes and records to a repository with retention locks and controlled audit access if immutable audit is required.

## 2. Event intake and RPA

Define a custom authenticated inbound service for source system, event ID, TOR technical key, UTC event timestamp, event type and source payload hash. Allowed business events: delivery exception, contractual transit delay, goods receipt quantity variance, temperature excursion. These names describe a custom contract, not SAP-delivered event constants.

On the first delivery, atomically insert event, case and evidence-collection outbox command. Repeated ID plus matching payload returns the existing case; repeated ID with a different payload is a conflict. A worker dispatches committed commands. Delivery is at least once; deduplication is mandatory.

OData V2 alone does not establish an event subscription or a universal change-notification channel. Select the actual TM event enhancement/PPF action or integration event publisher available in your system, and have it write a durable outbox in the owning transaction. Register no speculative SAP enhancement names. Avoid outbound network calls while holding a TM business-object lock.

RPA retrieves BOL, POD, photographs and telemetry, scans attachments, hashes bytes and writes documents to an authorized repository. Return repository identifiers and hashes; do not permit arbitrary callback URLs or client-supplied filesystem paths. Bot retries use the same command ID. Bots collect evidence; trusted SAP-side logic builds the mandatory entity manifest from versioned configuration.

## 3. TM and inference boundary

The supplied reader retrieves the TOR root. Add verified BOPF associations or supported services for item/load and receipt facts, planned/actual stops, execution events, carrier BP/SCAC mapping and agreement references. Resolve contract and SLA from your configured source; historical claims require a separate governed store. Do not update SAP TM tables directly.

Retrieve current authoritative data immediately before analysis, preserving an immutable snapshot with its revision and event-time contract. Before approving/posting, recheck the relevant revision; invalidate the decision if the evidence, contract, TOR facts or policy changed. Read carrier history only from claims adjudicated before the triggering event time.

An inference request should carry schema version, claim/version, request ID, snapshot hash, evidence hashes and approved feature schema version. The response must echo those identifiers and provide extracted entities, three class probabilities, calibrated precedent availability/probability, separate extractor/classifier/precedent model hashes, calibration version, top-five SHAP values and the feature snapshot. The core's `model_hash` can hold a canonical manifest hash covering those versions.

The adapter must validate JSON types, finite numeric values, ranges, lengths, model allowlist, expected feature names/count, explanation structure and correlation identifiers before constructing `ty_claim`. Presence of nonempty strings in the demo core is not full provenance validation. Never deserialize an untrusted request straight into authoritative `ty_claim` fields. Reject mismatched or expired results and route to a recoverable technical exception queue.

Use configured HTTPS destinations/certificates and bounded timeouts; store no credentials in ABAP source. Keep binary evidence outside decision JSON. OCR, RoBERTa, ResNet, fusion inference, cosine embeddings and SHAP run in an external inference service, not in this ABAP engine. Choose BTP or an approved on-premise service according to the actual architecture.

## 4. Rules and settlement calculation

Implement a versioned BRFplus or equivalent rule service returning SATISFIED, OPEN, AMBIGUOUS or EXCLUDED with the complete clause trace. OPEN requires an explicit contractual policy permitting precedent. Evaluate loading responsibility, custody, notification limits, exclusions and SLA rules from actual contract configuration; no generic hardcoded carrier-liability rule can substitute for this.

Compute settlement from approved valuation, coverage, deductible, limits, salvage and tax treatment in the finance/contract service. Normalize quantity and weight units before comparing. Round monetary amounts according to the target currency's SAP decimal configuration before constructing posting lines. The core checks positive amounts and settlement <= requested value; it does not calculate legal entitlement or account assignments.

## 5. Finance: standalone TM to ERP

The supplied class belongs in ERP and is only the inner staging step. Choose either a direct accounting document or the configured debit-memo-request/billing process. An FI document is not an SD debit memo request. Do not invoke both paths for the same claim. A carrier may be a supplier, customer or both; master data and accounting policy determine AR/AP and posting signs.

For direct FI, construct balanced G/L and AR/AP items and currency amounts, with approved company code, document type, dates and account assignments. The sample wrapper covers basic G/L + AR/AP documents; tax, CO-PA, extension fields and special processes need backend-specific additions. Pass the same token and payload hash on every retry.

Backend service protocol:

1. Authenticate the caller and authorize company, action and claim scope. Validate the persisted approved decision/version and all posting data. Do not accept a bot's claimed approval.
2. Acquire an enqueue lock on the ERP posting token. Read its durable status. Return the existing verified document for an already completed request; reject the same token with different payload bytes/hash.
3. Run `ZCL_AFC_FI_STAGE=>STAGE`. Any failed check/post, missing key or exception requires `BAPI_TRANSACTION_ROLLBACK` in this isolated service LUW.
4. Coordinate the posting-token completion record with FI's V1 update transaction (an appropriately designed update function), so a failed FI update cannot leave a false success marker. Validate this behavior for the actual BAPI implementation. A simple direct INSERT before COMMIT is insufficient proof of atomicity with V1 FI updates.
5. The transaction owner calls `BAPI_TRANSACTION_COMMIT` with WAIT = 'X', checks its return and verifies the actual accounting document plus token record. Return POSTED only after that verification. Preserve failed/unknown outcomes for reconciliation.
6. If TM times out after ERP may have committed, query by the same token. Never create a new token or blindly repost. `REF_DOC_NO` alone is not a uniqueness guarantee.
7. Persist confirmed ERP document identifiers in TM and append an audit transition. Reconcile pending/unknown transactions using a background job.

No distributed ACID transaction is assumed across standalone TM and ERP. A durable outbox plus ERP-side idempotency and reconciliation is required. The wrapper intentionally does not commit because only the outer service can coordinate that protocol.

## 6. Human review and appeals

Use application authorization checks at each service/action, not just menu access. Map permissions for intake, review, approve, post and appeal to scoped roles. Enforce optimistic case-version checks and lock state transitions. Record overrides as new decisions with reason, actor, old/new versions and independent review where required.

Calculate the 15-business-day appeal window using the agreed SAP factory calendar, timezone and notice date. A blind review UI must withhold the automated decision until the independent reviewer saves a judgment. Appeals against posted decisions initiate the backend's supported reversal/adjustment workflow; changing a claim status does not reverse accounting. Preserve original postings and decision versions. Only finalized reviewed outcomes enter the training corpus.

## Remaining landscape inputs

ERP product/release and SAP_BASIS level; TM support package; event producer; freight-order category and associations; authoritative contract/SLA and receipt sources; evidence and inference platform; company/currency policy; carrier account roles; direct FI versus SD debit memo flow; account/tax mappings; authorization roles; appeal calendar; durable audit repository. These inputs are needed to turn the reference package into deployable integration.
