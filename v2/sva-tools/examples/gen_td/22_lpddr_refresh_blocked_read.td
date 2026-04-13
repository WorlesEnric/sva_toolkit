diagram cmd_start__bundle_26 {
  clock posedge clk;
  disable iff !rst_n;

  param CMD_READY_MAX;
  param CORE_READY_MAX;
  param DATA_READY_MAX;
  param DATA_START_MAX;
  param REF_BLOCK_MAX;
  param RESP_READY_MAX;

  lane CMD_VALID: bit;
  lane REF_ACK: bit;
  lane REF_PENDING: bit;
  lane CMD_READY: bit;
  lane CORE_RD_READY: bit;
  lane CORE_RD_VALID: bit;
  lane DQ_READY: bit;
  lane DQ_VALID: bit;
  lane DQ_LAST: bit;
  lane RESP_READY: bit;
  lane RESP_VALID: bit;
  lane CMD_ADDR: bus;
  lane CMD_LEN: bus;
  lane CMD_TYPE: bus;
  lane CORE_ADDR: bus;
  lane CORE_ID: bus;
  lane CORE_LEN: bus;
  lane DQ_DATA: bus;
  lane DQ_ID: bus;
  lane RESP_ID: bus;
  lane RESP_STATUS: bus;

  anchor cmd_start__bundle_26__node_0 = high(REF_PENDING) and high(REF_ACK);
  anchor cmd_start__bundle_26__point_1 = 1'b1;
  anchor cmd_start__bundle_26__node_2 = 1'b1;
  anchor cmd_start__bundle_26__node_3 = rise(CMD_VALID);
  anchor cmd_start__bundle_26__node_4 = rise(CMD_VALID);
  anchor cmd_start__bundle_26__node_5 = high(CMD_VALID) and high(CMD_READY);
  anchor cmd_start__bundle_26__node_6 = rise(CORE_RD_VALID);
  anchor cmd_start__bundle_26__node_7 = high(CORE_RD_VALID) and high(CORE_RD_READY);
  anchor cmd_start__bundle_26__node_8 = rise(DQ_VALID);
  anchor cmd_start__bundle_26__node_9 = high(DQ_VALID) and high(DQ_READY);
  anchor cmd_start__bundle_26__node_10 = high(DQ_VALID) and high(DQ_READY) and high(DQ_LAST);
  anchor cmd_start__bundle_26__node_11 = high(CMD_VALID) and high(CMD_READY);
  anchor cmd_start__bundle_26__node_12 = rise(DQ_VALID);
  anchor cmd_start__bundle_26__node_13 = rise(REF_PENDING);
  anchor cmd_start__bundle_26__node_14 = high(REF_PENDING) and high(REF_ACK);
  anchor cmd_start__bundle_26__node_15 = rise(RESP_VALID);
  anchor cmd_start__bundle_26__node_16 = high(RESP_VALID) and high(RESP_READY);
  anchor cmd_start__bundle_26__point_17 = 1'b1;
  anchor cmd_start__bundle_26__node_18 = high(DQ_VALID) and high(DQ_READY) and high(DQ_LAST);
  anchor cmd_start__bundle_26__node_19 = rise(RESP_VALID);

  window cmd_start__bundle_26__win_0 = between cmd_start__bundle_26__node_0 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_1 = between cmd_start__bundle_26__point_1 and cmd_start__bundle_26__node_2 [0:0];
  window cmd_start__bundle_26__win_2 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_3 [1:2];
  window cmd_start__bundle_26__win_3 = between cmd_start__bundle_26__node_4 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_4 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_5 [0:CMD_READY_MAX];
  window cmd_start__bundle_26__win_5 = between cmd_start__bundle_26__node_6 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_6 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_7 [0:CORE_READY_MAX];
  window cmd_start__bundle_26__win_7 = between cmd_start__bundle_26__node_8 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_8 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_9 [0:DATA_READY_MAX];
  window cmd_start__bundle_26__win_9 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_10 [0:8];
  window cmd_start__bundle_26__win_10 = between cmd_start__bundle_26__node_11 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_11 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_12 [1:DATA_START_MAX];
  window cmd_start__bundle_26__win_12 = between cmd_start__bundle_26__point_1 and cmd_start__bundle_26__node_5 [0:0];
  window cmd_start__bundle_26__win_13 = between cmd_start__bundle_26__point_1 and cmd_start__bundle_26__node_7 [0:0];
  window cmd_start__bundle_26__win_14 = between cmd_start__bundle_26__point_1 and cmd_start__bundle_26__node_10 [0:0];
  window cmd_start__bundle_26__win_15 = between cmd_start__bundle_26__node_13 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_16 = between cmd_start__bundle_26__point_1 and cmd_start__bundle_26__node_14 [0:0];
  window cmd_start__bundle_26__win_17 = between cmd_start__bundle_26__node_15 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_18 = between cmd_start__bundle_26__point_1 and cmd_start__bundle_26__node_16 [0:0];
  window cmd_start__bundle_26__win_19 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_20 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_14 [0:REF_BLOCK_MAX];
  window cmd_start__bundle_26__win_21 = between cmd_start__bundle_26__node_18 and cmd_start__bundle_26__point_1 [0:0];
  window cmd_start__bundle_26__win_22 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_19 [1:2];
  window cmd_start__bundle_26__win_23 = between cmd_start__bundle_26__node_2 and cmd_start__bundle_26__node_16 [0:RESP_READY_MAX];
  window cmd_start__bundle_26__win_24 = between cmd_start__bundle_26__point_1 and cmd_start__bundle_26__node_9 [0:0];
  window cmd_start__bundle_26__win_25 = between cmd_start__bundle_26__node_3 and cmd_start__bundle_26__node_4 [0:0];
  window cmd_start__bundle_26__win_26 = between cmd_start__bundle_26__node_3 and cmd_start__bundle_26__node_11 [0:0];
  window cmd_start__bundle_26__win_27 = between cmd_start__bundle_26__node_5 and cmd_start__bundle_26__node_11 [0:0];
  window cmd_start__bundle_26__win_28 = between cmd_start__bundle_26__node_5 and cmd_start__bundle_26__node_4 [0:0];
  window cmd_start__bundle_26__win_29 = between cmd_start__bundle_26__node_7 and cmd_start__bundle_26__node_6 [0:0];
  window cmd_start__bundle_26__win_30 = between cmd_start__bundle_26__node_9 and cmd_start__bundle_26__node_8 [0:0];
  window cmd_start__bundle_26__win_31 = between cmd_start__bundle_26__node_9 and cmd_start__bundle_26__node_18 [0:0];
  window cmd_start__bundle_26__win_32 = between cmd_start__bundle_26__node_10 and cmd_start__bundle_26__node_8 [0:0];
  window cmd_start__bundle_26__win_33 = between cmd_start__bundle_26__node_10 and cmd_start__bundle_26__node_18 [0:0];
  window cmd_start__bundle_26__win_34 = between cmd_start__bundle_26__node_12 and cmd_start__bundle_26__node_8 [0:0];
  window cmd_start__bundle_26__win_35 = between cmd_start__bundle_26__node_12 and cmd_start__bundle_26__node_18 [0:0];
  window cmd_start__bundle_26__win_36 = between cmd_start__bundle_26__node_14 and cmd_start__bundle_26__node_13 [0:0];
  window cmd_start__bundle_26__win_37 = between cmd_start__bundle_26__node_16 and cmd_start__bundle_26__node_15 [0:0];
  window cmd_start__bundle_26__win_38 = between cmd_start__bundle_26__node_19 and cmd_start__bundle_26__node_15 [0:0];
  window high_CMD_VALID_from_cmd_rise_until_cmd_hs__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_11 [0:$];
  window high_DQ_VALID_from_dq_rise_until_dq_last__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_18 [0:$];
  window high_REF_PENDING_from_ref_rise_until_ref_done__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_0 [0:$];
  window low_CMD_VALID_before_ref_done__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_0 [0:$];
  window low_RESP_VALID_before_dq_last__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_18 [0:$];
  window stable_CMD_ADDR_from_cmd_rise_until_cmd_hs__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_11 [0:$];
  window stable_CMD_LEN_from_cmd_rise_until_cmd_hs__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_11 [0:$];
  window stable_CMD_TYPE_from_cmd_rise_until_cmd_hs__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_11 [0:$];
  window stable_DQ_ID_from_dq_rise_until_dq_last__hold_2 = between cmd_start__bundle_26__point_17 and cmd_start__bundle_26__node_18 [0:$];

  show high(CMD_VALID) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_11;
  show high(DQ_VALID) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_18;
  show high(REF_PENDING) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_0;
  show low(CMD_VALID) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_0;
  show low(RESP_VALID) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_18;
  show stable(CMD_ADDR) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_11;
  show stable(CMD_LEN) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_11;
  show stable(CMD_TYPE) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_11;
  show stable(DQ_ID) from cmd_start__bundle_26__point_17 until cmd_start__bundle_26__node_18;

  property cmd_start:
    REF_PENDING && REF_ACK |-> ##[1:2] $rose(CMD_VALID);
  property cmd_wait:
    $rose(CMD_VALID) |-> ##[0:CMD_READY_MAX] CMD_VALID && CMD_READY;
  property core_wait:
    $rose(CORE_RD_VALID) |-> ##[0:CORE_READY_MAX] CORE_RD_VALID && CORE_RD_READY;
  property data_first_gap:
    $rose(DQ_VALID) |-> ##[0:DATA_READY_MAX] DQ_VALID && DQ_READY;
  property data_phase:
    $rose(DQ_VALID) |-> ##[0:8] DQ_VALID && DQ_READY && DQ_LAST;
  property data_start:
    CMD_VALID && CMD_READY |-> ##[1:DATA_START_MAX] $rose(DQ_VALID);
  property high_CMD_VALID_from_cmd_rise_until_cmd_hs:
    $rose(CMD_VALID) |-> CMD_VALID until_with CMD_VALID && CMD_READY;
  property high_CORE_RD_VALID_from_core_req_rise_until_core_req_hs:
    $rose(CORE_RD_VALID) |-> CORE_RD_VALID until_with CORE_RD_VALID && CORE_RD_READY;
  property high_DQ_VALID_from_dq_rise_until_dq_last:
    $rose(DQ_VALID) |-> DQ_VALID until_with DQ_VALID && DQ_READY && DQ_LAST;
  property high_REF_PENDING_from_ref_rise_until_ref_done:
    $rose(REF_PENDING) |-> REF_PENDING until_with REF_PENDING && REF_ACK;
  property high_RESP_VALID_from_resp_rise_until_resp_hs:
    $rose(RESP_VALID) |-> RESP_VALID until_with RESP_VALID && RESP_READY;
  property low_CMD_VALID_before_ref_done [lossy]:
    !CMD_VALID until REF_PENDING && REF_ACK;
  // low_CMD_VALID_before_ref_done: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_CMD_VALID_before_ref_done: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_CMD_VALID_before_ref_done: until excludes the terminating cycle and is shown as a hold-until skeleton
  property low_RESP_VALID_before_dq_last [lossy]:
    !RESP_VALID until DQ_VALID && DQ_READY && DQ_LAST;
  // low_RESP_VALID_before_dq_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RESP_VALID_before_dq_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RESP_VALID_before_dq_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  property refresh_block:
    $rose(REF_PENDING) |-> ##[0:REF_BLOCK_MAX] REF_PENDING && REF_ACK;
  property resp_start:
    DQ_VALID && DQ_READY && DQ_LAST |-> ##[1:2] $rose(RESP_VALID);
  property resp_wait:
    $rose(RESP_VALID) |-> ##[0:RESP_READY_MAX] RESP_VALID && RESP_READY;
  property stable_CMD_ADDR_from_cmd_rise_until_cmd_hs:
    $rose(CMD_VALID) |-> $stable(CMD_ADDR) until_with CMD_VALID && CMD_READY;
  property stable_CMD_LEN_from_cmd_rise_until_cmd_hs:
    $rose(CMD_VALID) |-> $stable(CMD_LEN) until_with CMD_VALID && CMD_READY;
  property stable_CMD_TYPE_from_cmd_rise_until_cmd_hs:
    $rose(CMD_VALID) |-> $stable(CMD_TYPE) until_with CMD_VALID && CMD_READY;
  property stable_CORE_ADDR_from_core_req_rise_until_core_req_hs:
    $rose(CORE_RD_VALID) |-> $stable(CORE_ADDR) until_with CORE_RD_VALID && CORE_RD_READY;
  property stable_CORE_ID_from_core_req_rise_until_core_req_hs:
    $rose(CORE_RD_VALID) |-> $stable(CORE_ID) until_with CORE_RD_VALID && CORE_RD_READY;
  property stable_CORE_LEN_from_core_req_rise_until_core_req_hs:
    $rose(CORE_RD_VALID) |-> $stable(CORE_LEN) until_with CORE_RD_VALID && CORE_RD_READY;
  property stable_DQ_DATA_from_dq_rise_until_dq_first:
    $rose(DQ_VALID) |-> $stable(DQ_DATA) until_with DQ_VALID && DQ_READY;
  property stable_DQ_ID_from_dq_rise_until_dq_last:
    $rose(DQ_VALID) |-> $stable(DQ_ID) until_with DQ_VALID && DQ_READY && DQ_LAST;
  property stable_RESP_ID_from_resp_rise_until_resp_hs:
    $rose(RESP_VALID) |-> $stable(RESP_ID) until_with RESP_VALID && RESP_READY;
  property stable_RESP_STATUS_from_resp_rise_until_resp_hs:
    $rose(RESP_VALID) |-> $stable(RESP_STATUS) until_with RESP_VALID && RESP_READY;
}