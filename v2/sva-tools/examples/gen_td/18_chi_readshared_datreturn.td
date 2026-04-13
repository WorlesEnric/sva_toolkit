diagram dat_first_wait__bundle_20 {
  clock posedge clk;
  disable iff !rst_n;

  param DAT_READY_MAX;
  param BURST_MAX;
  param DAT_START_MAX;
  param REQ_READY_MAX;
  param RSP_LAT_MAX;

  lane RXDAT_READY: bit;
  lane RXDAT_VALID: bit;
  lane RXDAT_LAST: bit;
  lane RXRSP_READY: bit;
  lane RXRSP_VALID: bit;
  lane TXREQ_READY: bit;
  lane TXREQ_VALID: bit;
  lane RXDAT_DATA: bus;
  lane RXDAT_RESP: bus;
  lane RXDAT_TXNID: bus;
  lane RXRSP_OPCODE: bus;
  lane RXRSP_TXNID: bus;
  lane TXREQ_ADDR: bus;
  lane TXREQ_OPCODE: bus;
  lane TXREQ_SRCID: bus;
  lane TXREQ_TXNID: bus;

  anchor dat_first_wait__bundle_20__node_0 = rise(RXDAT_VALID);
  anchor dat_first_wait__bundle_20__point_1 = 1'b1;
  anchor dat_first_wait__bundle_20__node_2 = 1'b1;
  anchor dat_first_wait__bundle_20__node_3 = high(RXDAT_VALID) and high(RXDAT_READY);
  anchor dat_first_wait__bundle_20__node_4 = high(RXDAT_VALID) and high(RXDAT_READY) and high(RXDAT_LAST);
  anchor dat_first_wait__bundle_20__node_5 = high(RXRSP_VALID) and high(RXRSP_READY);
  anchor dat_first_wait__bundle_20__node_6 = rise(RXDAT_VALID);
  anchor dat_first_wait__bundle_20__node_7 = high(RXDAT_VALID) and high(RXDAT_READY) and high(RXDAT_LAST);
  anchor dat_first_wait__bundle_20__node_8 = high(RXDAT_READY);
  anchor dat_first_wait__bundle_20__node_9 = rise(RXRSP_VALID);
  anchor dat_first_wait__bundle_20__node_10 = high(RXRSP_VALID) and high(RXRSP_READY);
  anchor dat_first_wait__bundle_20__node_11 = rise(TXREQ_VALID);
  anchor dat_first_wait__bundle_20__node_12 = high(TXREQ_VALID) and high(TXREQ_READY);
  anchor dat_first_wait__bundle_20__point_13 = 1'b1;
  anchor dat_first_wait__bundle_20__node_14 = high(TXREQ_VALID) and high(TXREQ_READY);
  anchor dat_first_wait__bundle_20__node_15 = rise(RXRSP_VALID);

  window dat_first_wait__bundle_20__win_0 = between dat_first_wait__bundle_20__node_0 and dat_first_wait__bundle_20__point_1 [0:0];
  window dat_first_wait__bundle_20__win_1 = between dat_first_wait__bundle_20__point_1 and dat_first_wait__bundle_20__node_2 [0:0];
  window dat_first_wait__bundle_20__win_2 = between dat_first_wait__bundle_20__node_2 and dat_first_wait__bundle_20__node_3 [0:DAT_READY_MAX];
  window dat_first_wait__bundle_20__win_3 = between dat_first_wait__bundle_20__node_2 and dat_first_wait__bundle_20__node_4 [0:BURST_MAX];
  window dat_first_wait__bundle_20__win_4 = between dat_first_wait__bundle_20__node_5 and dat_first_wait__bundle_20__point_1 [0:0];
  window dat_first_wait__bundle_20__win_5 = between dat_first_wait__bundle_20__node_2 and dat_first_wait__bundle_20__node_6 [1:DAT_START_MAX];
  window dat_first_wait__bundle_20__win_6 = between dat_first_wait__bundle_20__node_7 and dat_first_wait__bundle_20__point_1 [0:0];
  window dat_first_wait__bundle_20__win_7 = between dat_first_wait__bundle_20__point_1 and dat_first_wait__bundle_20__node_8 [0:0];
  window dat_first_wait__bundle_20__win_8 = between dat_first_wait__bundle_20__point_1 and dat_first_wait__bundle_20__node_4 [0:0];
  window dat_first_wait__bundle_20__win_9 = between dat_first_wait__bundle_20__node_9 and dat_first_wait__bundle_20__point_1 [0:0];
  window dat_first_wait__bundle_20__win_10 = between dat_first_wait__bundle_20__point_1 and dat_first_wait__bundle_20__node_10 [0:0];
  window dat_first_wait__bundle_20__win_11 = between dat_first_wait__bundle_20__node_11 and dat_first_wait__bundle_20__point_1 [0:0];
  window dat_first_wait__bundle_20__win_12 = between dat_first_wait__bundle_20__point_1 and dat_first_wait__bundle_20__node_12 [0:0];
  window dat_first_wait__bundle_20__win_13 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__point_1 [0:0];
  window dat_first_wait__bundle_20__win_14 = between dat_first_wait__bundle_20__node_2 and dat_first_wait__bundle_20__node_12 [0:REQ_READY_MAX];
  window dat_first_wait__bundle_20__win_15 = between dat_first_wait__bundle_20__node_14 and dat_first_wait__bundle_20__point_1 [0:0];
  window dat_first_wait__bundle_20__win_16 = between dat_first_wait__bundle_20__node_2 and dat_first_wait__bundle_20__node_15 [1:RSP_LAT_MAX];
  window dat_first_wait__bundle_20__win_17 = between dat_first_wait__bundle_20__node_2 and dat_first_wait__bundle_20__node_10 [0:1];
  window dat_first_wait__bundle_20__win_18 = between dat_first_wait__bundle_20__point_1 and dat_first_wait__bundle_20__node_3 [0:0];
  window dat_first_wait__bundle_20__win_19 = between dat_first_wait__bundle_20__node_3 and dat_first_wait__bundle_20__node_0 [0:0];
  window dat_first_wait__bundle_20__win_20 = between dat_first_wait__bundle_20__node_3 and dat_first_wait__bundle_20__node_7 [0:0];
  window dat_first_wait__bundle_20__win_21 = between dat_first_wait__bundle_20__node_4 and dat_first_wait__bundle_20__node_7 [0:0];
  window dat_first_wait__bundle_20__win_22 = between dat_first_wait__bundle_20__node_4 and dat_first_wait__bundle_20__node_0 [0:0];
  window dat_first_wait__bundle_20__win_23 = between dat_first_wait__bundle_20__node_6 and dat_first_wait__bundle_20__node_7 [0:0];
  window dat_first_wait__bundle_20__win_24 = between dat_first_wait__bundle_20__node_6 and dat_first_wait__bundle_20__node_0 [0:0];
  window dat_first_wait__bundle_20__win_25 = between dat_first_wait__bundle_20__node_10 and dat_first_wait__bundle_20__node_9 [0:0];
  window dat_first_wait__bundle_20__win_26 = between dat_first_wait__bundle_20__node_12 and dat_first_wait__bundle_20__node_11 [0:0];
  window dat_first_wait__bundle_20__win_27 = between dat_first_wait__bundle_20__node_12 and dat_first_wait__bundle_20__node_14 [0:0];
  window dat_first_wait__bundle_20__win_28 = between dat_first_wait__bundle_20__node_15 and dat_first_wait__bundle_20__node_9 [0:0];
  window high_RXDAT_VALID_from_dat_rise_until_dat_last__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_7 [0:$];
  window high_RXRSP_VALID_from_rsp_rise_until_rsp_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_5 [0:$];
  window high_TXREQ_VALID_from_req_issue_until_req_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_14 [0:$];
  window low_RXDAT_VALID_before_rsp_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_5 [0:$];
  window stable_RXDAT_RESP_from_dat_rise_until_dat_last__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_7 [0:$];
  window stable_RXDAT_TXNID_from_dat_rise_until_dat_last__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_7 [0:$];
  window stable_RXRSP_OPCODE_from_rsp_rise_until_rsp_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_5 [0:$];
  window stable_RXRSP_TXNID_from_rsp_rise_until_rsp_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_5 [0:$];
  window stable_TXREQ_ADDR_from_req_issue_until_req_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_14 [0:$];
  window stable_TXREQ_OPCODE_from_req_issue_until_req_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_14 [0:$];
  window stable_TXREQ_SRCID_from_req_issue_until_req_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_14 [0:$];
  window stable_TXREQ_TXNID_from_req_issue_until_req_accept__hold_2 = between dat_first_wait__bundle_20__point_13 and dat_first_wait__bundle_20__node_14 [0:$];

  show high(RXDAT_VALID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_7;
  show high(RXRSP_VALID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_5;
  show high(TXREQ_VALID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_14;
  show low(RXDAT_VALID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_5;
  show stable(RXDAT_RESP) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_7;
  show stable(RXDAT_TXNID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_7;
  show stable(RXRSP_OPCODE) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_5;
  show stable(RXRSP_TXNID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_5;
  show stable(TXREQ_ADDR) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_14;
  show stable(TXREQ_OPCODE) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_14;
  show stable(TXREQ_SRCID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_14;
  show stable(TXREQ_TXNID) from dat_first_wait__bundle_20__point_13 until dat_first_wait__bundle_20__node_14;

  property dat_first_wait:
    $rose(RXDAT_VALID) |-> ##[0:DAT_READY_MAX] RXDAT_VALID && RXDAT_READY;
  property dat_phase:
    $rose(RXDAT_VALID) |-> ##[0:BURST_MAX] RXDAT_VALID && RXDAT_READY && RXDAT_LAST;
  property dat_start:
    RXRSP_VALID && RXRSP_READY |-> ##[1:DAT_START_MAX] $rose(RXDAT_VALID);
  property high_RXDAT_READY_at_dat_last:
    RXDAT_VALID && RXDAT_READY && RXDAT_LAST |-> RXDAT_READY;
  property high_RXDAT_VALID_from_dat_rise_until_dat_last:
    $rose(RXDAT_VALID) |-> RXDAT_VALID until_with RXDAT_VALID && RXDAT_READY && RXDAT_LAST;
  property high_RXRSP_VALID_from_rsp_rise_until_rsp_accept:
    $rose(RXRSP_VALID) |-> RXRSP_VALID until_with RXRSP_VALID && RXRSP_READY;
  property high_TXREQ_VALID_from_req_issue_until_req_accept:
    $rose(TXREQ_VALID) |-> TXREQ_VALID until_with TXREQ_VALID && TXREQ_READY;
  property low_RXDAT_VALID_before_rsp_accept [lossy]:
    !RXDAT_VALID until RXRSP_VALID && RXRSP_READY;
  // low_RXDAT_VALID_before_rsp_accept: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RXDAT_VALID_before_rsp_accept: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RXDAT_VALID_before_rsp_accept: until excludes the terminating cycle and is shown as a hold-until skeleton
  property req_wait:
    $rose(TXREQ_VALID) |-> ##[0:REQ_READY_MAX] TXREQ_VALID && TXREQ_READY;
  property rsp_latency:
    TXREQ_VALID && TXREQ_READY |-> ##[1:RSP_LAT_MAX] $rose(RXRSP_VALID);
  property rsp_wait:
    $rose(RXRSP_VALID) |-> ##[0:1] RXRSP_VALID && RXRSP_READY;
  property stable_RXDAT_DATA_from_dat_rise_until_dat_first:
    $rose(RXDAT_VALID) |-> $stable(RXDAT_DATA) until_with RXDAT_VALID && RXDAT_READY;
  property stable_RXDAT_RESP_from_dat_rise_until_dat_last:
    $rose(RXDAT_VALID) |-> $stable(RXDAT_RESP) until_with RXDAT_VALID && RXDAT_READY && RXDAT_LAST;
  property stable_RXDAT_TXNID_from_dat_rise_until_dat_last:
    $rose(RXDAT_VALID) |-> $stable(RXDAT_TXNID) until_with RXDAT_VALID && RXDAT_READY && RXDAT_LAST;
  property stable_RXRSP_OPCODE_from_rsp_rise_until_rsp_accept:
    $rose(RXRSP_VALID) |-> $stable(RXRSP_OPCODE) until_with RXRSP_VALID && RXRSP_READY;
  property stable_RXRSP_TXNID_from_rsp_rise_until_rsp_accept:
    $rose(RXRSP_VALID) |-> $stable(RXRSP_TXNID) until_with RXRSP_VALID && RXRSP_READY;
  property stable_TXREQ_ADDR_from_req_issue_until_req_accept:
    $rose(TXREQ_VALID) |-> $stable(TXREQ_ADDR) until_with TXREQ_VALID && TXREQ_READY;
  property stable_TXREQ_OPCODE_from_req_issue_until_req_accept:
    $rose(TXREQ_VALID) |-> $stable(TXREQ_OPCODE) until_with TXREQ_VALID && TXREQ_READY;
  property stable_TXREQ_SRCID_from_req_issue_until_req_accept:
    $rose(TXREQ_VALID) |-> $stable(TXREQ_SRCID) until_with TXREQ_VALID && TXREQ_READY;
  property stable_TXREQ_TXNID_from_req_issue_until_req_accept:
    $rose(TXREQ_VALID) |-> $stable(TXREQ_TXNID) until_with TXREQ_VALID && TXREQ_READY;
}