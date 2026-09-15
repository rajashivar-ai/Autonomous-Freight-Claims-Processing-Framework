REPORT zafc_tm_read.
" Read-only TM 9.6 BOPF example. Enter the technical root key, not TOR_ID.
" Confirm DDIC types and signatures in your installed support package.
PARAMETERS p_key TYPE /bobf/conf_key OBLIGATORY.

START-OF-SELECTION.
  DATA lt_keys TYPE /bobf/t_frw_key.
  DATA lt_root TYPE /scmtms/t_tor_root_k.
  DATA lt_failed TYPE /bobf/t_frw_key.
  DATA lo_message TYPE REF TO /bobf/if_frw_message.
  APPEND VALUE #( key = p_key ) TO lt_keys.
  TRY.
      DATA(lo_service) = /bobf/cl_tra_serv_mgr_factory=>get_service_manager(
        /scmtms/if_tor_c=>sc_bo_key ).
      lo_service->retrieve(
        EXPORTING iv_node_key = /scmtms/if_tor_c=>sc_node-root
                  it_key = lt_keys iv_fill_data = abap_true
        IMPORTING et_data = lt_root et_failed_key = lt_failed
                  eo_message = lo_message ).
      IF lt_failed IS NOT INITIAL OR lines( lt_root ) <> 1.
        WRITE / 'Freight order unavailable; inspect BOPF messages and authorization.'.
        RETURN.
      ENDIF.
      READ TABLE lt_root INDEX 1 INTO DATA(ls_root).
      WRITE: / 'Freight order:', ls_root-tor_id,
             / 'Technical key:', ls_root-key.
      " Next: map approved node associations for items, stops, execution,
      " business partners and freight agreement references in your system.
      " Contracts, SLA and historical outcomes are NOT all TOR root fields.
      " Do not infer a carrier SCAC from a BP identifier.
    CATCH /bobf/cx_frw INTO DATA(lx_bopf).
      WRITE / lx_bopf->get_text( ).
  ENDTRY.
