// Example 21: USB control endpoint read transfer on EP0
// Scenario:
//   A control SETUP packet is accepted after backpressure, the IN data stage
//   returns two packets with initial sink stall, and the status handshake is
//   held until the host acknowledges it.
// Reference properties:
//   1. (SETUP_VALID && !SETUP_READY) |-> (SETUP_VALID && $stable(BMREQ) &&
//      $stable(BREQ) && $stable(WVALUE) && $stable(WLEN)) until_with SETUP_READY
//   2. (DATA_VALID && !DATA_READY) |-> (DATA_VALID && $stable(DATA_PID) &&
//      $stable(DATA_BYTES) && $stable(DATA_COUNT)) until_with DATA_READY
//   3. (STATUS_VALID && !STATUS_READY) |-> (STATUS_VALID && $stable(STATUS_PID)) until_with STATUS_READY
//   4. !STATUS_VALID until (DATA_VALID && DATA_READY && DATA_LAST)

diagram usb3_ep0_control_read {
  clock posedge clk;
  disable iff !rst_n;
  ticks 20;

  param SETUP_READY_MAX;
  param DATA_START_MAX;
  param DATA_READY_MAX;
  param STATUS_READY_MAX;

  // SETUP stage
  lane SETUP_VALID: bit      = 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane SETUP_READY: bit      = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane BMREQ:       bus[8]   = x x 80 80 x x x x x x x x x x x x x x x x;
  lane BREQ:        bus[8]   = x x 06 06 x x x x x x x x x x x x x x x x;
  lane WVALUE:      bus[16]  = x x 0100 0100 x x x x x x x x x x x x x x x x;
  lane WLEN:        bus[16]  = x x 0012 0012 x x x x x x x x x x x x x x x x;

  // IN data stage
  lane DATA_VALID:  bit      = 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane DATA_READY:  bit      = 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane DATA_PID:    bus[8]   = x x x x x x IN IN IN x x x x x x x x x x x;
  lane DATA_BYTES:  bus[32]  = x x x x x x D0 D0 D1 x x x x x x x x x x x;
  lane DATA_COUNT:  bus[8]   = x x x x x x 12 12 06 x x x x x x x x x x x;
  lane DATA_LAST:   bit      = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;

  // STATUS stage
  lane STATUS_VALID: bit     = 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0;
  lane STATUS_READY: bit     = 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane STATUS_PID:   bus[8]  = x x x x x x x x x x x ACK ACK x x x x x x x;
  lane EP_BUSY:      bit     = 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0;

  //
  // Anchors — setup accept, data burst, status completion
  //
  anchor setup_rise   = rise(SETUP_VALID);
  anchor setup_hs     = high(SETUP_VALID) and high(SETUP_READY);
  anchor data_rise    = rise(DATA_VALID);
  anchor data_first   = high(DATA_VALID) and high(DATA_READY);
  anchor data_last    = high(DATA_VALID) and high(DATA_READY) and high(DATA_LAST);
  anchor status_rise  = rise(STATUS_VALID);
  anchor status_hs    = high(STATUS_VALID) and high(STATUS_READY);

  //
  // Windows — control transfer stages
  //
  window setup_wait     = between setup_rise and setup_hs [0:SETUP_READY_MAX];
  window data_start     = between setup_hs and data_rise [1:DATA_START_MAX];
  window data_first_gap = between data_rise and data_first [0:DATA_READY_MAX];
  window data_phase     = between data_rise and data_last [0:8];
  window status_start   = between data_last and status_rise [1:2];
  window status_wait    = between status_rise and status_hs [0:STATUS_READY_MAX];

  //
  // Cuts — remove idle bus intervals around the transfer
  //
  cut pre_setup   = before setup_rise omitted label "idle endpoint";
  cut post_status = after status_hs omitted label "ep0 idle";

  //
  // Constraints — setup, data, and status packets must hold until accepted
  //
  show high(SETUP_VALID) from setup_rise until setup_hs;
  show stable(BMREQ) from setup_rise until setup_hs;
  show stable(BREQ) from setup_rise until setup_hs;
  show stable(WVALUE) from setup_rise until setup_hs;
  show stable(WLEN) from setup_rise until setup_hs;

  show high(DATA_VALID) from data_rise until data_last;
  show stable(DATA_PID) from data_rise until data_last;
  show stable(DATA_BYTES) from data_rise until data_first;
  show stable(DATA_COUNT) from data_rise until data_last;

  show high(STATUS_VALID) from status_rise until status_hs;
  show stable(STATUS_PID) from status_rise until status_hs;
  show high(EP_BUSY) from setup_hs until status_hs;

  show low(DATA_VALID) before setup_hs;
  show low(STATUS_VALID) before data_last;
}
