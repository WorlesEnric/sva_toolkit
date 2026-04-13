// Example 22: LPDDR read request blocked by refresh, then command/data/response
// Scenario:
//   A core read request is accepted, an intervening refresh must complete before
//   the DRAM read command can be issued, read data returns with first-beat stall,
//   and the response channel finally wakes the core.
// Reference properties:
//   1. (CORE_RD_VALID && !CORE_RD_READY) |-> (CORE_RD_VALID && $stable(CORE_ADDR) &&
//      $stable(CORE_LEN) && $stable(CORE_ID)) until_with CORE_RD_READY
//   2. (REF_PENDING && !REF_ACK) |-> REF_PENDING until_with REF_ACK
//   3. (CMD_VALID && !CMD_READY) |-> (CMD_VALID && $stable(CMD_TYPE) &&
//      $stable(CMD_ADDR) && $stable(CMD_LEN)) until_with CMD_READY
//   4. (DQ_VALID && !DQ_READY) |-> (DQ_VALID && $stable(DQ_DATA) && $stable(DQ_ID)) until_with DQ_READY
//   5. (RESP_VALID && !RESP_READY) |-> (RESP_VALID && $stable(RESP_ID) && $stable(RESP_STATUS)) until_with RESP_READY

diagram lpddr_refresh_blocked_read {
  clock posedge clk;
  disable iff !rst_n;
  ticks 20;

  param CORE_READY_MAX;
  param REF_BLOCK_MAX;
  param CMD_READY_MAX;
  param DATA_START_MAX;
  param DATA_READY_MAX;
  param RESP_READY_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;

  // Core request
  lane CORE_RD_VALID: bit      = 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CORE_RD_READY: bit      = 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CORE_ADDR:     bus[32]  = x x A800 x x x x x x x x x x x x x x x x x;
  lane CORE_LEN:      bus[8]   = x x 03 x x x x x x x x x x x x x x x x x;
  lane CORE_ID:       bus[4]   = x x 7 x x x x x x x x x x x x x x x x x;

  // Refresh blocking window
  lane REF_PENDING:   bit      = 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane REF_ACK:       bit      = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  // DRAM command issue
  lane CMD_VALID:     bit      = 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CMD_READY:     bit      = 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CMD_TYPE:      bus[8]   = x x x x x x RD RD x x x x x x x x x x x x;
  lane CMD_ADDR:      bus[32]  = x x x x x x A800 A800 x x x x x x x x x x x x;
  lane CMD_LEN:       bus[8]   = x x x x x x 03 03 x x x x x x x x x x x x;

  // DRAM read data
  lane DQ_VALID:      bit      = 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0;
  lane DQ_READY:      bit      = 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0;
  lane DQ_DATA:       bus[128] = x x x x x x x x x L0 L0 L1 L2 x x x x x x x;
  lane DQ_ID:         bus[4]   = x x x x x x x x x 7 7 7 7 x x x x x x x;
  lane DQ_LAST:       bit      = 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;

  // Core response
  lane RESP_VALID:    bit      = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0;
  lane RESP_READY:    bit      = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
  lane RESP_ID:       bus[4]   = x x x x x x x x x x x x x x 7 7 x x x x;
  lane RESP_STATUS:   bus[8]   = x x x x x x x x x x x x x x OK OK x x x x;

  //
  // Anchors — request accept, refresh end, command issue, data, response
  //
  anchor core_req_rise = rise(CORE_RD_VALID);
  anchor core_req_hs   = high(CORE_RD_VALID) and high(CORE_RD_READY);
  anchor ref_rise      = rise(REF_PENDING);
  anchor ref_done      = high(REF_PENDING) and high(REF_ACK);
  anchor cmd_rise      = rise(CMD_VALID);
  anchor cmd_hs        = high(CMD_VALID) and high(CMD_READY);
  anchor dq_rise       = rise(DQ_VALID);
  anchor dq_first      = high(DQ_VALID) and high(DQ_READY);
  anchor dq_last       = high(DQ_VALID) and high(DQ_READY) and high(DQ_LAST);
  anchor resp_rise     = rise(RESP_VALID);
  anchor resp_hs       = high(RESP_VALID) and high(RESP_READY);

  //
  // Windows — refresh block and read-return path
  //
  window core_wait      = between core_req_rise and core_req_hs [0:CORE_READY_MAX];
  window refresh_block  = between ref_rise and ref_done [0:REF_BLOCK_MAX];
  window cmd_start      = between ref_done and cmd_rise [1:2];
  window cmd_wait       = between cmd_rise and cmd_hs [0:CMD_READY_MAX];
  window data_start     = between cmd_hs and dq_rise [1:DATA_START_MAX];
  window data_first_gap = between dq_rise and dq_first [0:DATA_READY_MAX];
  window data_phase     = between dq_rise and dq_last [0:8];
  window resp_start     = between dq_last and resp_rise [1:2];
  window resp_wait      = between resp_rise and resp_hs [0:RESP_READY_MAX];

  //
  // Cuts — compress unrelated cycles
  //
  cut pre_req    = before core_req_rise omitted label "no pending miss";
  cut post_resp  = after resp_hs omitted label "core wakes up";

  //
  // Constraints — each phase must hold payload until acceptance
  //
  show high(CORE_RD_VALID) from core_req_rise until core_req_hs;
  show stable(CORE_ADDR) from core_req_rise until core_req_hs;
  show stable(CORE_LEN) from core_req_rise until core_req_hs;
  show stable(CORE_ID) from core_req_rise until core_req_hs;

  show high(REF_PENDING) from ref_rise until ref_done;
  show low(CMD_VALID) before ref_done;

  show high(CMD_VALID) from cmd_rise until cmd_hs;
  show stable(CMD_TYPE) from cmd_rise until cmd_hs;
  show stable(CMD_ADDR) from cmd_rise until cmd_hs;
  show stable(CMD_LEN) from cmd_rise until cmd_hs;

  show high(DQ_VALID) from dq_rise until dq_last;
  show stable(DQ_DATA) from dq_rise until dq_first;
  show stable(DQ_ID) from dq_rise until dq_last;

  show high(RESP_VALID) from resp_rise until resp_hs;
  show stable(RESP_ID) from resp_rise until resp_hs;
  show stable(RESP_STATUS) from resp_rise until resp_hs;
  show low(RESP_VALID) before dq_last;
}
