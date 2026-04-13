// Example 13: AXI4 read channel burst with address backpressure
// Scenario:
//   A single AXI4 read burst is issued, the AR channel stalls for a few cycles,
//   the first R beat is also backpressured, and then the remaining beats retire.
// Reference properties:
//   1. (ARVALID && !ARREADY) |-> (ARVALID && $stable(ARADDR) && $stable(ARLEN) &&
//      $stable(ARSIZE) && $stable(ARBURST) && $stable(ARID)) until_with ARREADY
//   2. (RVALID && !RREADY) |-> ($stable(RDATA) && $stable(RRESP) && $stable(RID) && RVALID) until_with RREADY
//   3. $rose(ARVALID) |-> ##[1:R_START_MAX] $rose(RVALID)
//   4. !$rose(RVALID) until (ARVALID && ARREADY)

diagram axi4_read_channel {
  clock posedge ACLK;
  disable iff !ARESETn;
  ticks 18;

  param AR_READY_MAX;
  param R_START_MAX;
  param R_READY_MAX;
  param BURST_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;
  param ID_WIDTH;

  // Address channel
  lane ARVALID: bit     = 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ARREADY: bit     = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ARADDR:  bus[32] = x x RA00 RA00 x x x x x x x x x x x x x x;
  lane ARLEN:   bus[8]  = x x 03 03 x x x x x x x x x x x x x x;
  lane ARSIZE:  bus[3]  = x x 02 02 x x x x x x x x x x x x x x;
  lane ARBURST: bus[2]  = x x 01 01 x x x x x x x x x x x x x x;
  lane ARID:    bus[4]  = x x 4 4 x x x x x x x x x x x x x x;

  // Read data channel
  lane RVALID:  bit     = 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0;
  lane RREADY:  bit     = 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0;
  lane RDATA:   bus[32] = x x x x x D0 D0 D1 D2 x x x x x x x x x;
  lane RRESP:   bus[2]  = x x x x x 00 00 00 00 x x x x x x x x x;
  lane RID:     bus[4]  = x x x x x 4 4 4 4 x x x x x x x x x;
  lane RLAST:   bit     = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;

  //
  // Anchors — AR phase
  //
  anchor ar_valid_rise = rise(ARVALID);
  anchor ar_handshake  = high(ARVALID) and high(ARREADY);

  //
  // Anchors — R phase
  //
  anchor r_valid_rise  = rise(RVALID);
  anchor r_first_beat  = high(RVALID) and high(RREADY);
  anchor r_last_beat   = high(RVALID) and high(RREADY) and high(RLAST);

  //
  // Windows — latency and burst shape
  //
  window ar_wait        = between ar_valid_rise and ar_handshake [0:AR_READY_MAX];
  window r_start        = between ar_handshake and r_valid_rise [1:R_START_MAX];
  window r_first_wait   = between r_valid_rise and r_first_beat [0:R_READY_MAX];
  window r_burst_phase  = between r_valid_rise and r_last_beat [0:BURST_MAX];
  window r_complete     = between r_first_beat and r_last_beat [0:BURST_MAX];

  //
  // Cuts — hide idle and post-burst cycles
  //
  cut pre_read   = before ar_valid_rise omitted label "idle";
  cut post_read  = after r_last_beat omitted label "next burst";

  //
  // Constraints — address stability and data-phase behavior
  //
  show high(ARVALID) from ar_valid_rise until ar_handshake;
  show stable(ARADDR) from ar_valid_rise until ar_handshake;
  show stable(ARLEN) from ar_valid_rise until ar_handshake;
  show stable(ARSIZE) from ar_valid_rise until ar_handshake;
  show stable(ARBURST) from ar_valid_rise until ar_handshake;
  show stable(ARID) from ar_valid_rise until ar_handshake;

  show high(RVALID) from r_valid_rise until r_last_beat;
  show stable(RDATA) from r_valid_rise until r_first_beat;
  show stable(RRESP) from r_valid_rise until r_last_beat;
  show stable(RID) from r_valid_rise until r_last_beat;

  show low(RVALID) before ar_handshake;
  show high(RREADY) at r_last_beat;
}
