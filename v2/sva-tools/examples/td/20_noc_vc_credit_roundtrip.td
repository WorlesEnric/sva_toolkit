// Example 20: NoC virtual-channel request, response burst, and credit return
// Scenario:
//   A request flit is injected into a virtual channel, stalls on the ingress
//   port, a response burst later arrives with initial backpressure, and credit
//   return information is sent back to reopen the virtual channel.
// Reference properties:
//   1. (TX_VALID && !TX_READY) |-> (TX_VALID && $stable(TX_VC) && $stable(TX_DEST) &&
//      $stable(TX_HDR) && $stable(TX_CLASS)) until_with TX_READY
//   2. (RX_VALID && !RX_READY) |-> (RX_VALID && $stable(RX_DATA) && $stable(RX_VC) &&
//      $stable(RX_CLASS)) until_with RX_READY
//   3. (CR_VALID && !CR_READY) |-> (CR_VALID && $stable(CR_VC) && $stable(CR_COUNT)) until_with CR_READY
//   4. !RX_VALID until (TX_VALID && TX_READY)

diagram noc_vc_credit_roundtrip {
  clock posedge clk;
  disable iff !rst_n;
  ticks 20;

  param TX_READY_MAX;
  param RX_START_MAX;
  param RX_READY_MAX;
  param CREDIT_LAT_MAX;
  param PAYLOAD_WIDTH;

  // Request injection
  lane TX_VALID: bit      = 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_READY: bit      = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_VC:    bus[4]   = x x 1 1 x x x x x x x x x x x x x x x x;
  lane TX_DEST:  bus[8]   = x x 2C 2C x x x x x x x x x x x x x x x x;
  lane TX_HDR:   bus[16]  = x x A100 A100 x x x x x x x x x x x x x x x x;
  lane TX_CLASS: bus[4]   = x x RQ RQ x x x x x x x x x x x x x x x x;

  // Response burst
  lane RX_VALID: bit      = 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0;
  lane RX_READY: bit      = 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0;
  lane RX_DATA:  bus[64]  = x x x x x x x x P0 P0 P1 P2 x x x x x x x x;
  lane RX_VC:    bus[4]   = x x x x x x x x 1 1 1 1 x x x x x x x x;
  lane RX_CLASS: bus[4]   = x x x x x x x x RS RS RS RS x x x x x x x x;
  lane RX_TAIL:  bit      = 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;

  // Credit return
  lane CR_VALID: bit      = 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0;
  lane CR_READY: bit      = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0;
  lane CR_VC:    bus[4]   = x x x x x x x x x x x x x 1 1 x x x x x;
  lane CR_COUNT: bus[8]   = x x x x x x x x x x x x x 02 02 x x x x x;
  lane VC_OPEN:  bit      = 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1;

  //
  // Anchors — request acceptance, response burst, credit return
  //
  anchor tx_rise    = rise(TX_VALID);
  anchor tx_hs      = high(TX_VALID) and high(TX_READY);
  anchor rx_rise    = rise(RX_VALID);
  anchor rx_first   = high(RX_VALID) and high(RX_READY);
  anchor rx_last    = high(RX_VALID) and high(RX_READY) and high(RX_TAIL);
  anchor cr_rise    = rise(CR_VALID);
  anchor cr_hs      = high(CR_VALID) and high(CR_READY);

  //
  // Windows — request/response/credit lifecycle
  //
  window tx_wait       = between tx_rise and tx_hs [0:TX_READY_MAX];
  window rx_start      = between tx_hs and rx_rise [1:RX_START_MAX];
  window rx_first_wait = between rx_rise and rx_first [0:RX_READY_MAX];
  window rx_phase      = between rx_rise and rx_last [0:8];
  window cr_latency    = between rx_last and cr_rise [0:CREDIT_LAT_MAX];
  window cr_wait       = between cr_rise and cr_hs [0:1];

  //
  // Cuts — hide unrelated mesh activity
  //
  cut pre_tx   = before tx_rise omitted label "router idle";
  cut post_cr  = after cr_hs omitted label "vc reopened";

  //
  // Constraints — VC metadata must hold until acceptance
  //
  show high(TX_VALID) from tx_rise until tx_hs;
  show stable(TX_VC) from tx_rise until tx_hs;
  show stable(TX_DEST) from tx_rise until tx_hs;
  show stable(TX_HDR) from tx_rise until tx_hs;
  show stable(TX_CLASS) from tx_rise until tx_hs;

  show high(RX_VALID) from rx_rise until rx_last;
  show stable(RX_DATA) from rx_rise until rx_first;
  show stable(RX_VC) from rx_rise until rx_last;
  show stable(RX_CLASS) from rx_rise until rx_last;

  show high(CR_VALID) from cr_rise until cr_hs;
  show stable(CR_VC) from cr_rise until cr_hs;
  show stable(CR_COUNT) from cr_rise until cr_hs;

  show low(RX_VALID) before tx_hs;
  show high(VC_OPEN) after cr_hs;
}
