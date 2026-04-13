diagram ar_wait__bundle_17 {
  clock posedge ACLK;
  disable iff !ARESETn;

  param AR_READY_MAX;
  param BURST_MAX;
  param R_READY_MAX;
  param R_START_MAX;

  lane ARREADY: bit;
  lane ARVALID: bit;
  lane RLAST: bit;
  lane RREADY: bit;
  lane RVALID: bit;
  lane ARADDR: bus;
  lane ARBURST: bus;
  lane ARID: bus;
  lane ARLEN: bus;
  lane ARSIZE: bus;
  lane RDATA: bus;
  lane RID: bus;
  lane RRESP: bus;

  anchor ar_wait__bundle_17__node_0 = rise(ARVALID);
  anchor ar_wait__bundle_17__point_1 = 1'b1;
  anchor ar_wait__bundle_17__node_2 = 1'b1;
  anchor ar_wait__bundle_17__node_3 = high(ARVALID) and high(ARREADY);
  anchor ar_wait__bundle_17__node_4 = high(RVALID) and high(RREADY) and high(RLAST);
  anchor ar_wait__bundle_17__node_5 = high(RREADY);
  anchor ar_wait__bundle_17__node_6 = rise(RVALID);
  anchor ar_wait__bundle_17__node_7 = high(RVALID) and high(RREADY) and high(RLAST);
  anchor ar_wait__bundle_17__point_8 = 1'b1;
  anchor ar_wait__bundle_17__node_9 = high(RVALID) and high(RREADY);
  anchor ar_wait__bundle_17__node_10 = high(RVALID) and high(RREADY);
  anchor ar_wait__bundle_17__node_11 = high(ARVALID) and high(ARREADY);
  anchor ar_wait__bundle_17__node_12 = rise(RVALID);

  window ar_wait__bundle_17__win_0 = between ar_wait__bundle_17__node_0 and ar_wait__bundle_17__point_1 [0:0];
  window ar_wait__bundle_17__win_1 = between ar_wait__bundle_17__point_1 and ar_wait__bundle_17__node_2 [0:0];
  window ar_wait__bundle_17__win_2 = between ar_wait__bundle_17__node_2 and ar_wait__bundle_17__node_3 [0:AR_READY_MAX];
  window ar_wait__bundle_17__win_3 = between ar_wait__bundle_17__point_1 and ar_wait__bundle_17__node_3 [0:0];
  window ar_wait__bundle_17__win_4 = between ar_wait__bundle_17__node_4 and ar_wait__bundle_17__point_1 [0:0];
  window ar_wait__bundle_17__win_5 = between ar_wait__bundle_17__point_1 and ar_wait__bundle_17__node_5 [0:0];
  window ar_wait__bundle_17__win_6 = between ar_wait__bundle_17__node_6 and ar_wait__bundle_17__point_1 [0:0];
  window ar_wait__bundle_17__win_7 = between ar_wait__bundle_17__point_1 and ar_wait__bundle_17__node_7 [0:0];
  window ar_wait__bundle_17__win_8 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__point_1 [0:0];
  window ar_wait__bundle_17__win_9 = between ar_wait__bundle_17__node_2 and ar_wait__bundle_17__node_7 [0:BURST_MAX];
  window ar_wait__bundle_17__win_10 = between ar_wait__bundle_17__node_9 and ar_wait__bundle_17__point_1 [0:0];
  window ar_wait__bundle_17__win_11 = between ar_wait__bundle_17__node_2 and ar_wait__bundle_17__node_10 [0:R_READY_MAX];
  window ar_wait__bundle_17__win_12 = between ar_wait__bundle_17__node_11 and ar_wait__bundle_17__point_1 [0:0];
  window ar_wait__bundle_17__win_13 = between ar_wait__bundle_17__node_2 and ar_wait__bundle_17__node_12 [1:R_START_MAX];
  window ar_wait__bundle_17__win_14 = between ar_wait__bundle_17__point_1 and ar_wait__bundle_17__node_10 [0:0];
  window ar_wait__bundle_17__win_15 = between ar_wait__bundle_17__node_3 and ar_wait__bundle_17__node_0 [0:0];
  window ar_wait__bundle_17__win_16 = between ar_wait__bundle_17__node_3 and ar_wait__bundle_17__node_11 [0:0];
  window ar_wait__bundle_17__win_17 = between ar_wait__bundle_17__node_5 and ar_wait__bundle_17__node_9 [0:0];
  window ar_wait__bundle_17__win_18 = between ar_wait__bundle_17__node_7 and ar_wait__bundle_17__node_6 [0:0];
  window ar_wait__bundle_17__win_19 = between ar_wait__bundle_17__node_7 and ar_wait__bundle_17__node_9 [0:0];
  window ar_wait__bundle_17__win_20 = between ar_wait__bundle_17__node_10 and ar_wait__bundle_17__node_6 [0:0];
  window ar_wait__bundle_17__win_21 = between ar_wait__bundle_17__node_12 and ar_wait__bundle_17__node_6 [0:0];
  window high_ARVALID_from_ar_valid_rise_until_ar_handshake__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_11 [0:$];
  window high_RVALID_from_r_valid_rise_until_r_last_beat__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_4 [0:$];
  window low_RVALID_before_ar_handshake__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_11 [0:$];
  window stable_ARADDR_from_ar_valid_rise_until_ar_handshake__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_11 [0:$];
  window stable_ARBURST_from_ar_valid_rise_until_ar_handshake__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_11 [0:$];
  window stable_ARID_from_ar_valid_rise_until_ar_handshake__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_11 [0:$];
  window stable_ARLEN_from_ar_valid_rise_until_ar_handshake__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_11 [0:$];
  window stable_ARSIZE_from_ar_valid_rise_until_ar_handshake__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_11 [0:$];
  window stable_RDATA_from_r_valid_rise_until_r_first_beat__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_9 [0:$];
  window stable_RID_from_r_valid_rise_until_r_last_beat__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_4 [0:$];
  window stable_RRESP_from_r_valid_rise_until_r_last_beat__hold_2 = between ar_wait__bundle_17__point_8 and ar_wait__bundle_17__node_4 [0:$];

  show high(ARVALID) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_11;
  show high(RVALID) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_4;
  show low(RVALID) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_11;
  show stable(ARADDR) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_11;
  show stable(ARBURST) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_11;
  show stable(ARID) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_11;
  show stable(ARLEN) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_11;
  show stable(ARSIZE) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_11;
  show stable(RDATA) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_9;
  show stable(RID) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_4;
  show stable(RRESP) from ar_wait__bundle_17__point_8 until ar_wait__bundle_17__node_4;

  property ar_wait:
    $rose(ARVALID) |-> ##[0:AR_READY_MAX] ARVALID && ARREADY;
  property high_ARVALID_from_ar_valid_rise_until_ar_handshake:
    $rose(ARVALID) |-> ARVALID until_with ARVALID && ARREADY;
  property high_RREADY_at_r_last_beat:
    RVALID && RREADY && RLAST |-> RREADY;
  property high_RVALID_from_r_valid_rise_until_r_last_beat:
    $rose(RVALID) |-> RVALID until_with RVALID && RREADY && RLAST;
  property low_RVALID_before_ar_handshake [lossy]:
    !RVALID until ARVALID && ARREADY;
  // low_RVALID_before_ar_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RVALID_before_ar_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RVALID_before_ar_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  property r_burst_phase:
    $rose(RVALID) |-> ##[0:BURST_MAX] RVALID && RREADY && RLAST;
  property r_complete:
    RVALID && RREADY |-> ##[0:BURST_MAX] RVALID && RREADY && RLAST;
  property r_first_wait:
    $rose(RVALID) |-> ##[0:R_READY_MAX] RVALID && RREADY;
  property r_start:
    ARVALID && ARREADY |-> ##[1:R_START_MAX] $rose(RVALID);
  property stable_ARADDR_from_ar_valid_rise_until_ar_handshake:
    $rose(ARVALID) |-> $stable(ARADDR) until_with ARVALID && ARREADY;
  property stable_ARBURST_from_ar_valid_rise_until_ar_handshake:
    $rose(ARVALID) |-> $stable(ARBURST) until_with ARVALID && ARREADY;
  property stable_ARID_from_ar_valid_rise_until_ar_handshake:
    $rose(ARVALID) |-> $stable(ARID) until_with ARVALID && ARREADY;
  property stable_ARLEN_from_ar_valid_rise_until_ar_handshake:
    $rose(ARVALID) |-> $stable(ARLEN) until_with ARVALID && ARREADY;
  property stable_ARSIZE_from_ar_valid_rise_until_ar_handshake:
    $rose(ARVALID) |-> $stable(ARSIZE) until_with ARVALID && ARREADY;
  property stable_RDATA_from_r_valid_rise_until_r_first_beat:
    $rose(RVALID) |-> $stable(RDATA) until_with RVALID && RREADY;
  property stable_RID_from_r_valid_rise_until_r_last_beat:
    $rose(RVALID) |-> $stable(RID) until_with RVALID && RREADY && RLAST;
  property stable_RRESP_from_r_valid_rise_until_r_last_beat:
    $rose(RVALID) |-> $stable(RRESP) until_with RVALID && RREADY && RLAST;
}