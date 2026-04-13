// Example 15: TileLink Get / GrantData burst with D-channel backpressure
// Scenario:
//   A TileLink A-channel Get request waits for ready, the D-channel response
//   starts later, its first beat is backpressured, and a multi-beat GrantData
//   burst completes with D_LAST asserted on the final beat.
// Reference properties:
//   1. (A_VALID && !A_READY) |-> (A_VALID && $stable(A_OPCODE) && $stable(A_ADDRESS) &&
//      $stable(A_SIZE) && $stable(A_SOURCE)) until_with A_READY
//   2. (D_VALID && !D_READY) |-> (D_VALID && $stable(D_DATA) && $stable(D_OPCODE) &&
//      $stable(D_SOURCE) && $stable(D_DENIED) && $stable(D_CORRUPT)) until_with D_READY
//   3. $rose(A_VALID) |-> ##[1:D_START_MAX] $rose(D_VALID)
//   4. !D_VALID until (A_VALID && A_READY)

diagram tilelink_get_grantdata {
  clock posedge clk;
  disable iff !rst_n;
  ticks 18;

  param A_READY_MAX;
  param D_START_MAX;
  param D_READY_MAX;
  param BURST_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;
  param SOURCE_WIDTH;

  // A channel
  lane A_VALID:   bit     = 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane A_READY:   bit     = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane A_OPCODE:  bus[3]  = x x 4 4 x x x x x x x x x x x x x x;
  lane A_ADDRESS: bus[32] = x x C800 C800 x x x x x x x x x x x x x x;
  lane A_SIZE:    bus[4]  = x x 06 06 x x x x x x x x x x x x x x;
  lane A_SOURCE:  bus[4]  = x x 2 2 x x x x x x x x x x x x x x;

  // D channel
  lane D_VALID:   bit     = 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0;
  lane D_READY:   bit     = 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0;
  lane D_OPCODE:  bus[3]  = x x x x x 1 1 1 1 x x x x x x x x x;
  lane D_SOURCE:  bus[4]  = x x x x x 2 2 2 2 x x x x x x x x x;
  lane D_DATA:    bus[64] = x x x x x Q0 Q0 Q1 Q2 x x x x x x x x x;
  lane D_DENIED:  bus[1]  = x x x x x 0 0 0 0 x x x x x x x x x;
  lane D_CORRUPT: bus[1]  = x x x x x 0 0 0 0 x x x x x x x x x;
  lane D_LAST:    bit     = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;

  //
  // Anchors — request and response phases
  //
  anchor a_get_rise   = rise(A_VALID);
  anchor a_handshake  = high(A_VALID) and high(A_READY);
  anchor d_valid_rise = rise(D_VALID);
  anchor d_first_beat = high(D_VALID) and high(D_READY);
  anchor d_last_beat  = high(D_VALID) and high(D_READY) and high(D_LAST);

  //
  // Windows — latency and burst shape
  //
  window a_wait        = between a_get_rise and a_handshake [0:A_READY_MAX];
  window d_start       = between a_handshake and d_valid_rise [1:D_START_MAX];
  window d_first_wait  = between d_valid_rise and d_first_beat [0:D_READY_MAX];
  window d_burst_phase = between d_valid_rise and d_last_beat [0:BURST_MAX];
  window d_completion  = between d_first_beat and d_last_beat [0:BURST_MAX];

  //
  // Cuts — show only the visible request/response interval
  //
  cut pre_get     = before a_get_rise omitted label "fabric idle";
  cut post_grant  = after d_last_beat omitted label "grant consumed";

  //
  // Constraints — A channel payload and D burst metadata must hold
  //
  show high(A_VALID) from a_get_rise until a_handshake;
  show stable(A_OPCODE) from a_get_rise until a_handshake;
  show stable(A_ADDRESS) from a_get_rise until a_handshake;
  show stable(A_SIZE) from a_get_rise until a_handshake;
  show stable(A_SOURCE) from a_get_rise until a_handshake;

  show high(D_VALID) from d_valid_rise until d_last_beat;
  show stable(D_DATA) from d_valid_rise until d_first_beat;
  show stable(D_OPCODE) from d_valid_rise until d_last_beat;
  show stable(D_SOURCE) from d_valid_rise until d_last_beat;
  show stable(D_DENIED) from d_valid_rise until d_last_beat;
  show stable(D_CORRUPT) from d_valid_rise until d_last_beat;

  show low(D_VALID) before a_handshake;
  show high(D_READY) at d_last_beat;
}
