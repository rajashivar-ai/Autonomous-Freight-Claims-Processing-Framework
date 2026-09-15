# Autonomous Freight Claims — SAP TM ABAP reference package

## Target and status

Prepared for **standalone SAP TM 9.6**, as requested, with ABAP 7.50 syntax as the compatibility target. The ERP finance backend/release is not yet specified. This is newly generated reference code based on sections 4 and 5.4 of the supplied manuscript; it is not the original evaluated implementation and does not reproduce its reported results.

The decision engine is implemented. Live event ingestion, RPA/OCR/ML services, persistent workflow, authorization configuration, appeals UI and backend-specific financial mapping require implementation in your landscape. No SAP connection was available: these sources have not been activated, compiled or executed in SAP. Included ABAP Unit tests are ready for execution there.

## Files and use

| File | Purpose | Run location |
|---|---|---|
| `src/zafc_claim_demo.prog.abap` | Decision engine, synthetic fixture, 12 ABAP Unit tests and executable demo | TM development system |
| `integration/zafc_tm_read.prog.abap` | BOPF TOR root retrieval by technical key | TM 9.6 development system |
| `integration/zcl_afc_fi_stage.clas.abap` | CHECK/POST wrapper for an already mapped accounting document; no commit | ERP finance development system |
| `INTEGRATION.md` | Event, evidence, inference, persistence, posting and appeal contracts | Implementation guide |

1. Create executable program `ZAFC_CLAIM_DEMO` in SE38 or ADT and paste the report source.
2. Run syntax check and activate. Run its local ABAP Unit tests through ADT or SE80.
3. Execute the report. Expected synthetic output: `APPROVED`, `DAMAGE`, `RULES`, USD 410. No posting occurs.
4. Create `ZAFC_TM_READ` separately in TM. Confirm the BOPF signatures against your installed support package. Supply a TOR technical root key, not a freight order number.
5. Create `ZCL_AFC_FI_STAGE` separately in the finance backend. It must be called only by a trusted backend service implementing the transaction protocol in `INTEGRATION.md`.
6. For application use, move the local engine and types into global Z classes/interfaces; preserve the supplied tests. The one-report packaging makes the core easier to inspect without DDIC setup.

## Implemented policy

| Check | Value / behavior |
|---|---|
| Freight order ID / SCAC | Exact nonempty match |
| Quantity / weight / timestamp similarity | 0.98 / 0.95 / 0.90 |
| Damage descriptor cosine similarity | 0.82 |
| Document agreement | Every document must reach 0.88 |
| Classifier | Valid three-class probability distribution; maximum >= 0.80 |
| Liability | SATISFIED contractual rules, or explicitly OPEN rules plus available precedent >= 0.70 |
| Sparse history | Fewer than 30 prior adjudicated claims disables precedent |
| Contract ambiguity/exclusion | Human review |
| Cap | Requested claim value above USD 25,000 requires review |
| Other currencies | Review until an approved currency/cap policy exists |
| Outcome | APPROVED or REVIEW; no automatic rejection |

The source thresholds are represented by policy version `PAPER_V1_USD`. They are research-reference settings, not approved business configuration. A production version should load versioned policy and entity manifests from protected customizing, scoped to company, carrier, contract and effective date. Do not accept policy, master data or rule results directly from a bot/client request.

### Interpretation decisions

- The manuscript mixes standalone TM 9.6 and S/4HANA 2022. This package uses the user's standalone TM selection. SAP lists TM 9.6 as based on NetWeaver 7.5; embedded TM is a separate deployment option. [SAP TM product information](https://help.sap.com/docs/PRODUCT_ID/b50cdc1d0fed4c1e855b8199f4e7e69c/3e3aae424f7d4a998ff8f79b38c962cc.html)
- Equation 7 is broader than the ambiguity safeguard in section 4.6. `OPEN` explicitly permits precedent; silent, conflicting or unmapped terms are `AMBIGUOUS` and go to review. `EXCLUDED` also goes to review, rather than allowing precedent to override an exclusion.
- Gate order is input/snapshot checks, entity validation, per-document agreement, classifier, contractual/precedent liability, then policy cap. A cap cannot authorize an otherwise failing case. The first failed stage is retained.
- The numerical formula defines equal zero values as similarity 1 and clamps negative similarities to zero. All numeric entities also require an absolute contractual tolerance, an additional conservative check for weight and timestamps.
- Timestamp inputs are UTC seconds from the same shipment baseline, with a separate absolute tolerance. Using calendar-number timestamps or Unix epoch ratios could conceal meaningful delays. This normalization is an explicit implementation choice requiring business validation.
- Validate documents against the **corresponding fact**: BOL load quantity against TM load quantity; POD receipt quantity against the authoritative receipt fact when available. A POD shortfall relative to load quantity is a classification feature, not a transcription error. Missing receipt truth requires review. Do not suppress shortage claims by comparing every quantity to the load.
- Model weights, the full 21-feature schema, contractual rule catalog, settlement valuation formula and supplementary corpus were not attached. The code accepts externally computed predictions and an independently calculated settlement amount; it does not fabricate these missing components.
- The manuscript's author notes about publishing a repository and reproducing evaluation figures were treated as reference text, not instructions to publish anything.

## Validation

Twelve ABAP Unit scenarios cover rules with sparse history, the precedent/classifier boundaries, sparse-history suppression, contract ambiguity, cap boundary, first-failure ordering, zero numeric values, absolute tolerance, invalid probabilities, missing evidence, per-document agreement and duplicate entities. They have **not run in SAP**.

Before production, add integration tests for authorization, concurrent duplicate events, stale snapshot/model responses, malformed and oversized payloads, remote timeout after posting, update-task failure, currency decimal precision, account/tax assignments, evidence integrity, outbox recovery and appeal reversal. The demo is intentionally synthetic; its hashes and explanation JSON are placeholders, not proof of model provenance.

## SAP references

- [SAP TM 9.x Enhancement Guide — BOPF consumer patterns](https://help.sap.com/doc/c7dab057b713442294901f1f7cbe069d/9.5.0/en-US/TM9xEnhancementGuide_2017.pdf). This is 9.x guidance; validate signatures in the installed 9.6 system.
- [SAP TM 9.6 configuration guide — TOR service-manager example](https://help.sap.com/doc/PRODUCTION/ffab464f0c654de48dccaaca2d373374/9.6.0/en-US/loioc4af8a260e3741788e0b86694e646aff_00024390.pdf).
- [SAP BAPI transaction model](https://help.sap.com/saphelp_gbt10/helpdata/en/4d/5b102ba1483d8fe10000000a42189e/content.htm?no_cache=true). The transaction owner controls commit/rollback.
- [SAP documentation for accounting-document checking](https://help.sap.com/docs/SAP_PROFITABILITY_PERFORMANCE_MANAGEMENT/b7ac37094c8043eb82340894a8f0006c/9d49fd9c986c4c31a3a1f2ce2d6e6ae4.html?locale=en-US&state=PRODUCTION&version=3.19). Finance availability and interface details must be checked in the selected backend.
