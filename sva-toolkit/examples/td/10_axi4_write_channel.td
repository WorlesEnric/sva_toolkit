// Example 10: AXI4 write channel protocol (most complex)
// Demonstrates: all DSL features together — many signals, parameters, events,
//               chained windows, cuts, multiple constraint regions, multiple
//               properties with exact/lossy/unsupported status, legacy rules

diagram axi4_write_channel {
  clock posedge ACLK;
  disable iff !ARESETn;
  ticks 16;

  param AW_READY_MAX;
  param W_READY_MAX;
  param BRESP_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;

  // Address channel
  lane AWVALID: bit     = 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane AWREADY: bit     = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane AWADDR:  bus[32] = x x A0 A0 A0 x x x x x x x x x x x;
  lane AWLEN:   bus[8]  = x x 02 02 02 x x x x x x x x x x x;
  lane AWSIZE:  bus[3]  = x x 02 02 02 x x x x x x x x x x x;
  lane AWBURST: bus[2]  = x x 01 01 01 x x x x x x x x x x x;

  // Write data channel
  lane WVALID:  bit     = 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0;
  lane WREADY:  bit     = 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0;
  lane WDATA:   bus[32] = x x x x x D0 D0 D1 D2 x x x x x x x;
  lane WSTRB:   bus[4]  = x x x x x 0F 0F 0F 0F x x x x x x x;
  lane WLAST:   bit     = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;

  // Write response channel
  lane BVALID:  bit     = 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0;
  lane BREADY:  bit     = 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
  lane BRESP:   bus[2]  = x x x x x x x x x x x 00 00 x x x;

  //
  // Anchors — address phase
  //
  anchor aw_valid_rise  = rise(AWVALID);
  anchor aw_handshake   = high(AWVALID) and high(AWREADY);

  //
  // Anchors — write data phase
  //
  anchor w_first_beat   = rise(WVALID);
  anchor w_beat         = high(WVALID) and high(WREADY);
  anchor w_last_beat    = high(WVALID) and high(WREADY) and high(WLAST);

  //
  // Anchors — write response phase
  //
  anchor b_valid_rise   = rise(BVALID);
  anchor b_handshake    = high(BVALID) and high(BREADY);

  //
  // Windows — timing relationships between phases
  //
  window aw_latency     = between aw_valid_rise and aw_handshake [0:AW_READY_MAX];
  window w_start        = between aw_handshake and w_first_beat [0:4];
  window w_data_phase   = between w_first_beat and w_last_beat [0:128];
  window w_beat_ready   = between w_first_beat and w_beat [0:W_READY_MAX];
  window b_latency      = between w_last_beat and b_valid_rise [1:BRESP_MAX];
  window b_handshake_w  = between b_valid_rise and b_handshake [0:16];

  //
  // Cuts — focus the diagram on the interesting region
  //
  cut pre_transaction  = before aw_valid_rise omitted label "idle / reset";
  cut post_response    = after b_handshake omitted label "next transaction";

  //
  // Constraints — signal behavior within phases
  //

  // Address channel: AWVALID must stay high until AWREADY
  show high(AWVALID) from aw_valid_rise until aw_handshake;
  show stable(AWADDR) from aw_valid_rise until aw_handshake;
  show stable(AWLEN) from aw_valid_rise until aw_handshake;
  show stable(AWSIZE) from aw_valid_rise until aw_handshake;
  show stable(AWBURST) from aw_valid_rise until aw_handshake;

  // Write data channel: WVALID must stay high until WREADY per beat
  show high(WVALID) from w_first_beat until w_last_beat;
  show stable(WDATA) from w_first_beat until w_beat;
  show stable(WSTRB) from w_first_beat until w_beat;

  // Write response: BVALID must stay high until BREADY
  show high(BVALID) from b_valid_rise until b_handshake;
  show stable(BRESP) from b_valid_rise until b_handshake;

  // Ordering: no data before address
  show low(WVALID) before aw_handshake;
}
