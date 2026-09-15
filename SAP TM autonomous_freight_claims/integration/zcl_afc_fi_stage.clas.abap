" Create global class ZCL_AFC_FI_STAGE in the ERP finance backend.
" This stages a pre-mapped accounting document in the caller's SAP LUW.
" It does NOT implement claim authorization, mapping, idempotency or commit.
CLASS zcl_afc_fi_stage DEFINITION PUBLIC FINAL CREATE PUBLIC.
  PUBLIC SECTION.
    TYPES tt_gl TYPE STANDARD TABLE OF bapiacgl09 WITH DEFAULT KEY.
    TYPES tt_ar TYPE STANDARD TABLE OF bapiacar09 WITH DEFAULT KEY.
    TYPES tt_ap TYPE STANDARD TABLE OF bapiacap09 WITH DEFAULT KEY.
    TYPES tt_amount TYPE STANDARD TABLE OF bapiaccr09 WITH DEFAULT KEY.
    TYPES tt_return TYPE STANDARD TABLE OF bapiret2 WITH DEFAULT KEY.
    CLASS-METHODS stage
      IMPORTING is_header TYPE bapiache09
                it_gl TYPE tt_gl it_ar TYPE tt_ar it_ap TYPE tt_ap
                it_amount TYPE tt_amount
      EXPORTING ev_staged TYPE abap_bool ev_obj_type TYPE bapiache09-obj_type
                ev_obj_key TYPE bapiache09-obj_key
                ev_obj_sys TYPE bapiache09-obj_sys et_return TYPE tt_return.
ENDCLASS.

CLASS zcl_afc_fi_stage IMPLEMENTATION.
  METHOD stage.
    DATA lt_gl TYPE tt_gl.
    DATA lt_ar TYPE tt_ar.
    DATA lt_ap TYPE tt_ap.
    DATA lt_amount TYPE tt_amount.
    DATA lt_check TYPE tt_return.
    DATA lt_post TYPE tt_return.
    CLEAR: ev_staged, ev_obj_type, ev_obj_key, ev_obj_sys, et_return.
    lt_gl = it_gl.
    lt_ar = it_ar.
    lt_ap = it_ap.
    lt_amount = it_amount.
    CALL FUNCTION 'BAPI_ACC_DOCUMENT_CHECK'
      EXPORTING documentheader = is_header
      TABLES accountgl = lt_gl accountreceivable = lt_ar
             accountpayable = lt_ap currencyamount = lt_amount
             return = lt_check.
    APPEND LINES OF lt_check TO et_return.
    LOOP AT lt_check INTO DATA(ls_return).
      IF ls_return-type CA 'AEX'.
        RETURN.
      ENDIF.
    ENDLOOP.
    " Restore inputs in case a checking implementation modified a table.
    lt_gl = it_gl.
    lt_ar = it_ar.
    lt_ap = it_ap.
    lt_amount = it_amount.
    CALL FUNCTION 'BAPI_ACC_DOCUMENT_POST'
      EXPORTING documentheader = is_header
      IMPORTING obj_type = ev_obj_type obj_key = ev_obj_key
                obj_sys = ev_obj_sys
      TABLES accountgl = lt_gl accountreceivable = lt_ar
             accountpayable = lt_ap currencyamount = lt_amount
             return = lt_post.
    APPEND LINES OF lt_post TO et_return.
    LOOP AT lt_post INTO ls_return.
      IF ls_return-type CA 'AEX'.
        CLEAR: ev_obj_type, ev_obj_key, ev_obj_sys.
        RETURN. " Caller MUST roll back this LUW.
      ENDIF.
    ENDLOOP.
    ev_staged = xsdbool( ev_obj_key IS NOT INITIAL ).
    " Caller must also roll back if no object key was returned.
    " Only a successful commit followed by document verification is POSTED.
  ENDMETHOD.
ENDCLASS.
