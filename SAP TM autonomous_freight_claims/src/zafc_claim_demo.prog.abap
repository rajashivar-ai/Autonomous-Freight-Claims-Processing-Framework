REPORT zafc_claim_demo.
" Standalone reference decision engine: ABAP 7.50 syntax target.
" No database writes, remote calls or financial postings in this report.

INTERFACE lif_types.
  TYPES: BEGIN OF ty_entity,
           document_id TYPE string,
           entity_id TYPE string,
           kind TYPE c LENGTH 12,
           present TYPE abap_bool,
           extracted_text TYPE string,
           master_text TYPE string,
           extracted_number TYPE decfloat34,
           master_number TYPE decfloat34,
           cosine TYPE decfloat34,
           tolerance TYPE decfloat34,
         END OF ty_entity,
         tt_entity TYPE STANDARD TABLE OF ty_entity WITH DEFAULT KEY.
  " The trusted adapter supplies a complete, policy-required entity list.
  " Timestamps: UTC seconds relative to the SAME shipment baseline,
  " with contractual absolute tolerance in seconds (not packed YYYYMMDD).
  TYPES: BEGIN OF ty_claim,
           claim_id TYPE string,
           event_id TYPE string,
           freight_order_id TYPE string,
           snapshot_id TYPE string,
           evidence_hash TYPE string,
           model_hash TYPE string,
           policy_version TYPE string,
           rule_trace TYPE string,
           feature_snapshot TYPE string,
           shap_top_five TYPE string,
           currency TYPE c LENGTH 3,
           requested_amount TYPE decfloat34,
           settlement_amount TYPE decfloat34,
           master_current TYPE abap_bool,
           manifest_complete TYPE abap_bool,
           contract_state TYPE c LENGTH 10,
           history_count TYPE i,
           precedent_available TYPE abap_bool,
           precedent TYPE decfloat34,
           p_damage TYPE decfloat34,
           p_delay TYPE decfloat34,
           p_shortage TYPE decfloat34,
           entities TYPE tt_entity,
         END OF ty_claim.
  " contract_state = SATISFIED, OPEN, AMBIGUOUS, EXCLUDED.
  " OPEN means an explicitly permitted precedent path, not missing clauses.
  TYPES: BEGIN OF ty_check,
           document_id TYPE string,
           entity_id TYPE string,
           similarity TYPE decfloat34,
           threshold TYPE decfloat34,
           variance TYPE decfloat34,
           passed TYPE abap_bool,
         END OF ty_check,
         tt_check TYPE STANDARD TABLE OF ty_check WITH DEFAULT KEY,
         BEGIN OF ty_document,
           document_id TYPE string,
           total TYPE decfloat34,
           count TYPE i,
           agreement TYPE decfloat34,
         END OF ty_document,
         tt_document TYPE SORTED TABLE OF ty_document
           WITH UNIQUE KEY document_id,
         BEGIN OF ty_decision,
           claim_id TYPE string,
           status TYPE c LENGTH 16,
           first_failure TYPE c LENGTH 30,
           approval_path TYPE c LENGTH 12,
           cause TYPE c LENGTH 8,
           confidence TYPE decfloat34,
           min_agreement TYPE decfloat34,
           amount TYPE decfloat34,
           currency TYPE c LENGTH 3,
           actor TYPE syuname,
           decided_at TYPE timestampl,
           input_snapshot TYPE ty_claim,
           checks TYPE tt_check,
           documents TYPE tt_document,
         END OF ty_decision.
ENDINTERFACE.

CLASS lcl_engine DEFINITION FINAL.
  PUBLIC SECTION.
    METHODS decide IMPORTING is_claim TYPE lif_types=>ty_claim
                   RETURNING VALUE(rs_result) TYPE lif_types=>ty_decision.
  PRIVATE SECTION.
    METHODS check_entity IMPORTING is_entity TYPE lif_types=>ty_entity
                         RETURNING VALUE(rs_check) TYPE lif_types=>ty_check.
ENDCLASS.

CLASS lcl_engine IMPLEMENTATION.
  METHOD check_entity.
    DATA lv_denominator TYPE decfloat34.
    rs_check-document_id = is_entity-document_id.
    rs_check-entity_id = is_entity-entity_id.
    IF is_entity-present <> abap_true.
      RETURN.
    ENDIF.
    CASE is_entity-kind.
      WHEN 'ORDER_ID' OR 'SCAC'.
        rs_check-threshold = 1.
        IF is_entity-extracted_text IS NOT INITIAL AND
           is_entity-extracted_text = is_entity-master_text.
          rs_check-similarity = 1.
        ENDIF.
      WHEN 'QUANTITY' OR 'WEIGHT' OR 'TIMESTAMP'.
        CASE is_entity-kind.
          WHEN 'QUANTITY'. rs_check-threshold = '0.98'.
          WHEN 'WEIGHT'. rs_check-threshold = '0.95'.
          WHEN 'TIMESTAMP'. rs_check-threshold = '0.90'.
        ENDCASE.
        IF is_entity-tolerance < 0.
          RETURN.
        ENDIF.
        IF is_entity-kind <> 'TIMESTAMP' AND
           ( is_entity-extracted_number < 0 OR
             is_entity-master_number < 0 ).
          RETURN.
        ENDIF.
        rs_check-variance = is_entity-extracted_number -
                            is_entity-master_number.
        lv_denominator = abs( is_entity-extracted_number ).
        IF abs( is_entity-master_number ) > lv_denominator.
          lv_denominator = abs( is_entity-master_number ).
        ENDIF.
        IF lv_denominator = 0.
          rs_check-similarity = 1.
        ELSE.
          rs_check-similarity = 1 -
            abs( rs_check-variance ) / lv_denominator.
          IF rs_check-similarity < 0.
            rs_check-similarity = 0.
          ENDIF.
        ENDIF.
        IF abs( rs_check-variance ) > is_entity-tolerance.
          RETURN.
        ENDIF.
      WHEN 'DESCRIPTION'.
        rs_check-threshold = '0.82'.
        IF is_entity-extracted_text IS INITIAL OR
           is_entity-master_text IS INITIAL OR
           is_entity-cosine < -1 OR is_entity-cosine > 1.
          RETURN.
        ENDIF.
        rs_check-similarity = is_entity-cosine.
      WHEN OTHERS.
        RETURN.
    ENDCASE.
    rs_check-passed = xsdbool(
      rs_check-similarity >= rs_check-threshold ).
  ENDMETHOD.

  METHOD decide.
    DATA ls_check TYPE lif_types=>ty_check.
    DATA lv_failed TYPE abap_bool.
    DATA lv_sum TYPE decfloat34.
    DATA lt_seen TYPE HASHED TABLE OF string WITH UNIQUE KEY table_line.
    DATA lv_key TYPE string.
    rs_result-claim_id = is_claim-claim_id.
    rs_result-input_snapshot = is_claim.
    rs_result-status = 'REVIEW'.
    rs_result-actor = sy-uname.
    GET TIME STAMP FIELD rs_result-decided_at.
    " Technical input errors are separate from business gate statistics.
    rs_result-first_failure = 'INPUT_INVALID'.
    IF is_claim-claim_id IS INITIAL OR is_claim-event_id IS INITIAL OR
       is_claim-freight_order_id IS INITIAL OR
       is_claim-snapshot_id IS INITIAL OR
       is_claim-evidence_hash IS INITIAL OR is_claim-model_hash IS INITIAL OR
       is_claim-policy_version <> 'PAPER_V1_USD' OR
       is_claim-rule_trace IS INITIAL OR
       is_claim-feature_snapshot IS INITIAL OR
       is_claim-shap_top_five IS INITIAL OR is_claim-history_count < 0 OR
       is_claim-requested_amount <= 0 OR is_claim-settlement_amount <= 0 OR
       is_claim-settlement_amount > is_claim-requested_amount.
      RETURN.
    ENDIF.
    rs_result-first_failure = 'MASTER_STALE'.
    IF is_claim-master_current <> abap_true.
      RETURN.
    ENDIF.
    rs_result-first_failure = 'EVIDENCE_INCOMPLETE'.
    IF is_claim-manifest_complete <> abap_true OR
       is_claim-entities IS INITIAL.
      RETURN.
    ENDIF.

    " M2: inspect every required entity and each document separately.
    LOOP AT is_claim-entities INTO DATA(ls_entity).
      IF ls_entity-document_id IS INITIAL OR ls_entity-entity_id IS INITIAL.
        RETURN.
      ENDIF.
      " Length-prefixing avoids ambiguous composite key delimiters.
      lv_key = |{ strlen( ls_entity-document_id ) }:{ ls_entity-document_id }| &&
               |{ strlen( ls_entity-entity_id ) }:{ ls_entity-entity_id }|.
      INSERT lv_key INTO TABLE lt_seen.
      IF sy-subrc <> 0.
        rs_result-first_failure = 'DUPLICATE_ENTITY'.
        RETURN.
      ENDIF.
      ls_check = check_entity( ls_entity ).
      APPEND ls_check TO rs_result-checks.
      IF ls_check-passed <> abap_true.
        lv_failed = abap_true.
      ENDIF.
      READ TABLE rs_result-documents ASSIGNING FIELD-SYMBOL(<doc>)
        WITH TABLE KEY document_id = ls_entity-document_id.
      IF sy-subrc <> 0.
        INSERT VALUE #( document_id = ls_entity-document_id )
          INTO TABLE rs_result-documents ASSIGNING <doc>.
      ENDIF.
      <doc>-total = <doc>-total + ls_check-similarity.
      <doc>-count = <doc>-count + 1.
    ENDLOOP.
    rs_result-min_agreement = 1.
    LOOP AT rs_result-documents ASSIGNING <doc>.
      <doc>-agreement = <doc>-total / <doc>-count.
      IF <doc>-agreement < rs_result-min_agreement.
        rs_result-min_agreement = <doc>-agreement.
      ENDIF.
    ENDLOOP.
    rs_result-first_failure = 'ENTITY_VALIDATION'.
    IF lv_failed = abap_true.
      RETURN.
    ENDIF.
    rs_result-first_failure = 'DOCUMENT_AGREEMENT'.
    IF rs_result-min_agreement < CONV decfloat34( '0.88' ).
      RETURN.
    ENDIF.

    " M3: validate the distribution before selecting its maximum.
    rs_result-first_failure = 'AI_RESPONSE_INVALID'.
    IF is_claim-p_damage < 0 OR is_claim-p_damage > 1 OR
       is_claim-p_delay < 0 OR is_claim-p_delay > 1 OR
       is_claim-p_shortage < 0 OR is_claim-p_shortage > 1.
      RETURN.
    ENDIF.
    lv_sum = is_claim-p_damage + is_claim-p_delay + is_claim-p_shortage.
    IF abs( lv_sum - 1 ) > CONV decfloat34( '0.000001' ).
      RETURN.
    ENDIF.
    rs_result-cause = 'DAMAGE'.
    rs_result-confidence = is_claim-p_damage.
    IF is_claim-p_delay > rs_result-confidence.
      rs_result-cause = 'DELAY'.
      rs_result-confidence = is_claim-p_delay.
    ENDIF.
    IF is_claim-p_shortage > rs_result-confidence.
      rs_result-cause = 'SHORTAGE'.
      rs_result-confidence = is_claim-p_shortage.
    ENDIF.
    rs_result-first_failure = 'CLASSIFIER_CONFIDENCE'.
    IF rs_result-confidence < CONV decfloat34( '0.80' ).
      RETURN.
    ENDIF.

    " M4: explicit exclusion and ambiguity always require human review.
    rs_result-first_failure = 'CONTRACT_REVIEW'.
    CASE is_claim-contract_state.
      WHEN 'SATISFIED'.
        rs_result-approval_path = 'RULES'.
      WHEN 'OPEN'.
        rs_result-first_failure = 'SPARSE_HISTORY'.
        IF is_claim-history_count < 30.
          RETURN.
        ENDIF.
        rs_result-first_failure = 'PRECEDENT_UNAVAILABLE'.
        IF is_claim-precedent_available <> abap_true.
          RETURN.
        ENDIF.
        rs_result-first_failure = 'AI_RESPONSE_INVALID'.
        IF is_claim-precedent < 0 OR is_claim-precedent > 1.
          RETURN.
        ENDIF.
        rs_result-first_failure = 'LIABILITY_CONFIDENCE'.
        IF is_claim-precedent < CONV decfloat34( '0.70' ).
          RETURN.
        ENDIF.
        rs_result-approval_path = 'PRECEDENT'.
      WHEN OTHERS.
        RETURN.
    ENDCASE.
    " Source cap is USD; no implicit foreign-currency conversion.
    rs_result-first_failure = 'CURRENCY_POLICY'.
    IF is_claim-currency <> 'USD'.
      RETURN.
    ENDIF.
    rs_result-first_failure = 'POLICY_CAP'.
    IF is_claim-requested_amount > 25000.
      RETURN.
    ENDIF.
    CLEAR rs_result-first_failure.
    rs_result-status = 'APPROVED'.
    rs_result-amount = is_claim-settlement_amount.
    rs_result-currency = is_claim-currency.
    " APPROVED is a decision; it never means POSTED or SETTLED.
  ENDMETHOD.
ENDCLASS.

CLASS lcl_fixture DEFINITION FINAL.
  PUBLIC SECTION.
    CLASS-METHODS sample RETURNING VALUE(rs_claim) TYPE lif_types=>ty_claim.
ENDCLASS.
CLASS lcl_fixture IMPLEMENTATION.
  METHOD sample.
    rs_claim = VALUE #(
      claim_id = 'DEMO-CLAIM-001' event_id = 'DEMO-EVENT-001'
      freight_order_id = 'FO-10001' snapshot_id = 'SYNTHETIC-SNAPSHOT'
      evidence_hash = 'SYNTHETIC-NOT-A-REAL-HASH'
      model_hash = 'SYNTHETIC-NOT-A-TRAINED-MODEL'
      policy_version = 'PAPER_V1_USD'
      rule_trace = 'Synthetic contract: carrier responsible; no exclusion'
      feature_snapshot = '{"synthetic":true}'
      shap_top_five = '["synthetic-demo-only"]'
      currency = 'USD' requested_amount = 410 settlement_amount = 410
      master_current = abap_true manifest_complete = abap_true
      contract_state = 'SATISFIED' history_count = 12
      p_damage = '0.90' p_delay = '0.06' p_shortage = '0.04'
      entities = VALUE #(
        ( document_id = 'POD-1' entity_id = 'ORDER' kind = 'ORDER_ID'
          present = abap_true extracted_text = 'FO-10001'
          master_text = 'FO-10001' )
        ( document_id = 'POD-1' entity_id = 'CARRIER' kind = 'SCAC'
          present = abap_true extracted_text = 'TEST' master_text = 'TEST' )
        ( document_id = 'POD-1' entity_id = 'LOAD' kind = 'QUANTITY'
          present = abap_true extracted_number = 100 master_number = 100
          tolerance = 2 ) ) ).
  ENDMETHOD.
ENDCLASS.

CLASS ltc_engine DEFINITION FINAL FOR TESTING
  DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    DATA mo_engine TYPE REF TO lcl_engine.
    DATA ms_claim TYPE lif_types=>ty_claim.
    METHODS setup.
    METHODS rules_sparse FOR TESTING.
    METHODS precedent_boundary FOR TESTING.
    METHODS sparse_open FOR TESTING.
    METHODS ambiguous FOR TESTING.
    METHODS cap_boundary FOR TESTING.
    METHODS first_failure FOR TESTING.
    METHODS numeric_zero FOR TESTING.
    METHODS tolerance FOR TESTING.
    METHODS bad_distribution FOR TESTING.
    METHODS missing_evidence FOR TESTING.
    METHODS per_document FOR TESTING.
    METHODS duplicate_entity FOR TESTING.
ENDCLASS.
CLASS ltc_engine IMPLEMENTATION.
  METHOD setup.
    CREATE OBJECT mo_engine.
    ms_claim = lcl_fixture=>sample( ).
  ENDMETHOD.
  METHOD rules_sparse.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-status exp = 'APPROVED' ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-approval_path exp = 'RULES' ).
  ENDMETHOD.
  METHOD precedent_boundary.
    ms_claim-contract_state = 'OPEN'.
    ms_claim-history_count = 30.
    ms_claim-precedent_available = abap_true.
    ms_claim-precedent = '0.70'.
    ms_claim-p_damage = '0.80'.
    ms_claim-p_delay = '0.16'.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-status exp = 'APPROVED' ).
  ENDMETHOD.
  METHOD sparse_open.
    ms_claim-contract_state = 'OPEN'.
    ms_claim-history_count = 29.
    ms_claim-precedent_available = abap_true.
    ms_claim-precedent = 1.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'SPARSE_HISTORY' ).
  ENDMETHOD.
  METHOD ambiguous.
    ms_claim-contract_state = 'AMBIGUOUS'.
    ms_claim-history_count = 50.
    ms_claim-precedent_available = abap_true.
    ms_claim-precedent = 1.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'CONTRACT_REVIEW' ).
  ENDMETHOD.
  METHOD cap_boundary.
    ms_claim-requested_amount = 25000.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-status exp = 'APPROVED' ).
    ms_claim-requested_amount = '25000.01'.
    ls_result = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-first_failure exp = 'POLICY_CAP' ).
    cl_abap_unit_assert=>assert_initial( act = ls_result-amount ).
  ENDMETHOD.
  METHOD first_failure.
    ms_claim-entities[ 1 ]-extracted_text = 'WRONG'.
    ms_claim-p_damage = '0.70'.
    ms_claim-p_delay = '0.26'.
    ms_claim-requested_amount = 26000.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'ENTITY_VALIDATION' ).
  ENDMETHOD.
  METHOD numeric_zero.
    ms_claim-entities[ 3 ]-extracted_number = 0.
    ms_claim-entities[ 3 ]-master_number = 0.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-status exp = 'APPROVED' ).
  ENDMETHOD.
  METHOD tolerance.
    ms_claim-entities[ 3 ]-extracted_number = 99.
    ms_claim-entities[ 3 ]-tolerance = 0.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'ENTITY_VALIDATION' ).
  ENDMETHOD.
  METHOD bad_distribution.
    ms_claim-p_damage = '1.1'.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'AI_RESPONSE_INVALID' ).
  ENDMETHOD.
  METHOD missing_evidence.
    ms_claim-manifest_complete = abap_false.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'EVIDENCE_INCOMPLETE' ).
  ENDMETHOD.
  METHOD per_document.
    APPEND VALUE #( document_id = 'PHOTO-2' entity_id = 'DAMAGE'
      kind = 'DESCRIPTION' present = abap_true
      extracted_text = 'crushed carton' master_text = 'carton damage'
      cosine = '0.85' ) TO ms_claim-entities.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'DOCUMENT_AGREEMENT' ).
  ENDMETHOD.
  METHOD duplicate_entity.
    APPEND ms_claim-entities[ 1 ] TO ms_claim-entities.
    DATA(ls_result) = mo_engine->decide( ms_claim ).
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-first_failure exp = 'DUPLICATE_ENTITY' ).
  ENDMETHOD.
ENDCLASS.

START-OF-SELECTION.
  DATA(lo_engine) = NEW lcl_engine( ).
  DATA(ls_decision) = lo_engine->decide( lcl_fixture=>sample( ) ).
  WRITE: / 'SYNTHETIC DEMO - NO POSTING',
         / 'Claim:', ls_decision-claim_id,
         / 'Status:', ls_decision-status,
         / 'Root cause:', ls_decision-cause,
         / 'Approval path:', ls_decision-approval_path,
         / 'First failure:', ls_decision-first_failure,
         / 'Approved amount:', ls_decision-amount, ls_decision-currency.
