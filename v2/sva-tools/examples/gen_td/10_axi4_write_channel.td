diagram aw_latency__bundle_17 {
  clock posedge ACLK;
  disable iff !ARESETn;

  param AW_READY_MAX;
  param BRESP_MAX;
  param W_READY_MAX;

  lane AWREADY: bit;
  lane AWVALID: bit;
  lane BREADY: bit;
  lane BVALID: bit;
  lane WLAST: bit;
  lane WREADY: bit;
  lane WVALID: bit;
  lane AWADDR: bus;
  lane AWBURST: bus;
  lane AWLEN: bus;
  lane AWSIZE: bus;
  lane BRESP: bus;
  lane WDATA: bus;
  lane WSTRB: bus;

  anchor aw_latency__bundle_17__node_0 = rise(AWVALID);
  anchor aw_latency__bundle_17__point_1 = 1'b1;
  anchor aw_latency__bundle_17__node_2 = 1'b1;
  anchor aw_latency__bundle_17__node_3 = high(AWVALID) and high(AWREADY);
  anchor aw_latency__bundle_17__node_4 = rise(BVALID);
  anchor aw_latency__bundle_17__node_5 = high(BVALID) and high(BREADY);
  anchor aw_latency__bundle_17__node_6 = high(WVALID) and high(WREADY) and high(WLAST);
  anchor aw_latency__bundle_17__node_7 = rise(BVALID);
  anchor aw_latency__bundle_17__node_8 = rise(WVALID);
  anchor aw_latency__bundle_17__node_9 = high(WVALID) and high(WREADY) and high(WLAST);
  anchor aw_latency__bundle_17__point_10 = 1'b1;
  anchor aw_latency__bundle_17__node_11 = high(WVALID) and high(WREADY);
  anchor aw_latency__bundle_17__node_12 = high(AWVALID) and high(AWREADY);
  anchor aw_latency__bundle_17__node_13 = rise(WVALID);

  window aw_latency__bundle_17__win_0 = between aw_latency__bundle_17__node_0 and aw_latency__bundle_17__point_1 [0:0];
  window aw_latency__bundle_17__win_1 = between aw_latency__bundle_17__point_1 and aw_latency__bundle_17__node_2 [0:0];
  window aw_latency__bundle_17__win_2 = between aw_latency__bundle_17__node_2 and aw_latency__bundle_17__node_3 [0:AW_READY_MAX];
  window aw_latency__bundle_17__win_3 = between aw_latency__bundle_17__node_4 and aw_latency__bundle_17__point_1 [0:0];
  window aw_latency__bundle_17__win_4 = between aw_latency__bundle_17__node_2 and aw_latency__bundle_17__node_5 [0:16];
  window aw_latency__bundle_17__win_5 = between aw_latency__bundle_17__node_6 and aw_latency__bundle_17__point_1 [0:0];
  window aw_latency__bundle_17__win_6 = between aw_latency__bundle_17__node_2 and aw_latency__bundle_17__node_7 [1:BRESP_MAX];
  window aw_latency__bundle_17__win_7 = between aw_latency__bundle_17__point_1 and aw_latency__bundle_17__node_3 [0:0];
  window aw_latency__bundle_17__win_8 = between aw_latency__bundle_17__point_1 and aw_latency__bundle_17__node_5 [0:0];
  window aw_latency__bundle_17__win_9 = between aw_latency__bundle_17__node_8 and aw_latency__bundle_17__point_1 [0:0];
  window aw_latency__bundle_17__win_10 = between aw_latency__bundle_17__point_1 and aw_latency__bundle_17__node_9 [0:0];
  window aw_latency__bundle_17__win_11 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__point_1 [0:0];
  window aw_latency__bundle_17__win_12 = between aw_latency__bundle_17__point_1 and aw_latency__bundle_17__node_11 [0:0];
  window aw_latency__bundle_17__win_13 = between aw_latency__bundle_17__node_2 and aw_latency__bundle_17__node_11 [0:W_READY_MAX];
  window aw_latency__bundle_17__win_14 = between aw_latency__bundle_17__node_2 and aw_latency__bundle_17__node_9 [0:128];
  window aw_latency__bundle_17__win_15 = between aw_latency__bundle_17__node_12 and aw_latency__bundle_17__point_1 [0:0];
  window aw_latency__bundle_17__win_16 = between aw_latency__bundle_17__node_2 and aw_latency__bundle_17__node_13 [0:4];
  window aw_latency__bundle_17__win_17 = between aw_latency__bundle_17__node_3 and aw_latency__bundle_17__node_0 [0:0];
  window aw_latency__bundle_17__win_18 = between aw_latency__bundle_17__node_3 and aw_latency__bundle_17__node_12 [0:0];
  window aw_latency__bundle_17__win_19 = between aw_latency__bundle_17__node_5 and aw_latency__bundle_17__node_4 [0:0];
  window aw_latency__bundle_17__win_20 = between aw_latency__bundle_17__node_7 and aw_latency__bundle_17__node_4 [0:0];
  window aw_latency__bundle_17__win_21 = between aw_latency__bundle_17__node_9 and aw_latency__bundle_17__node_8 [0:0];
  window aw_latency__bundle_17__win_22 = between aw_latency__bundle_17__node_11 and aw_latency__bundle_17__node_8 [0:0];
  window high_AWVALID_from_aw_valid_rise_until_aw_handshake__hold_2 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__node_12 [0:$];
  window high_WVALID_from_w_first_beat_until_w_last_beat__hold_2 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__node_6 [0:$];
  window low_WVALID_before_aw_handshake__hold_2 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__node_12 [0:$];
  window stable_AWADDR_from_aw_valid_rise_until_aw_handshake__hold_2 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__node_12 [0:$];
  window stable_AWBURST_from_aw_valid_rise_until_aw_handshake__hold_2 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__node_12 [0:$];
  window stable_AWLEN_from_aw_valid_rise_until_aw_handshake__hold_2 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__node_12 [0:$];
  window stable_AWSIZE_from_aw_valid_rise_until_aw_handshake__hold_2 = between aw_latency__bundle_17__point_10 and aw_latency__bundle_17__node_12 [0:$];

  show high(AWVALID) from aw_latency__bundle_17__point_10 until aw_latency__bundle_17__node_12;
  show high(WVALID) from aw_latency__bundle_17__point_10 until aw_latency__bundle_17__node_6;
  show low(WVALID) from aw_latency__bundle_17__point_10 until aw_latency__bundle_17__node_12;
  show stable(AWADDR) from aw_latency__bundle_17__point_10 until aw_latency__bundle_17__node_12;
  show stable(AWBURST) from aw_latency__bundle_17__point_10 until aw_latency__bundle_17__node_12;
  show stable(AWLEN) from aw_latency__bundle_17__point_10 until aw_latency__bundle_17__node_12;
  show stable(AWSIZE) from aw_latency__bundle_17__point_10 until aw_latency__bundle_17__node_12;

  property aw_latency:
    $rose(AWVALID) |-> ##[0:AW_READY_MAX] AWVALID && AWREADY;
  property b_handshake_w:
    $rose(BVALID) |-> ##[0:16] BVALID && BREADY;
  property b_latency:
    WVALID && WREADY && WLAST |-> ##[1:BRESP_MAX] $rose(BVALID);
  property high_AWVALID_from_aw_valid_rise_until_aw_handshake:
    $rose(AWVALID) |-> AWVALID until_with AWVALID && AWREADY;
  property high_BVALID_from_b_valid_rise_until_b_handshake:
    $rose(BVALID) |-> BVALID until_with BVALID && BREADY;
  property high_WVALID_from_w_first_beat_until_w_last_beat:
    $rose(WVALID) |-> WVALID until_with WVALID && WREADY && WLAST;
  property low_WVALID_before_aw_handshake [lossy]:
    !WVALID until AWVALID && AWREADY;
  // low_WVALID_before_aw_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_WVALID_before_aw_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_WVALID_before_aw_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  property stable_AWADDR_from_aw_valid_rise_until_aw_handshake:
    $rose(AWVALID) |-> $stable(AWADDR) until_with AWVALID && AWREADY;
  property stable_AWBURST_from_aw_valid_rise_until_aw_handshake:
    $rose(AWVALID) |-> $stable(AWBURST) until_with AWVALID && AWREADY;
  property stable_AWLEN_from_aw_valid_rise_until_aw_handshake:
    $rose(AWVALID) |-> $stable(AWLEN) until_with AWVALID && AWREADY;
  property stable_AWSIZE_from_aw_valid_rise_until_aw_handshake:
    $rose(AWVALID) |-> $stable(AWSIZE) until_with AWVALID && AWREADY;
  property stable_BRESP_from_b_valid_rise_until_b_handshake:
    $rose(BVALID) |-> $stable(BRESP) until_with BVALID && BREADY;
  property stable_WDATA_from_w_first_beat_until_w_beat:
    $rose(WVALID) |-> $stable(WDATA) until_with WVALID && WREADY;
  property stable_WSTRB_from_w_first_beat_until_w_beat:
    $rose(WVALID) |-> $stable(WSTRB) until_with WVALID && WREADY;
  property w_beat_ready:
    $rose(WVALID) |-> ##[0:W_READY_MAX] WVALID && WREADY;
  property w_data_phase:
    $rose(WVALID) |-> ##[0:128] WVALID && WREADY && WLAST;
  property w_start:
    AWVALID && AWREADY |-> ##[0:4] $rose(WVALID);
}