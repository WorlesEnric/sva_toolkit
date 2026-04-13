// Example 18: CHI ReadShared request followed by response and data return
// Scenario:
//   A CHI requester issues a ReadShared transaction, the REQ channel stalls,
//   the RSP channel later acknowledges the request, and DAT returns a short
//   burst whose first beat is backpressured before completing.
// Reference properties:
//   1. (TXREQ_VALID && !TXREQ_READY) |-> (TXREQ_VALID && $stable(TXREQ_OPCODE) &&
//      $stable(TXREQ_ADDR) && $stable(TXREQ_SRCID) && $stable(TXREQ_TXNID)) until_with TXREQ_READY
//   2. (RXRSP_VALID && !RXRSP_READY) |-> (RXRSP_VALID && $stable(RXRSP_OPCODE) &&
//      $stable(RXRSP_TXNID)) until_with RXRSP_READY
//   3. (RXDAT_VALID && !RXDAT_READY) |-> (RXDAT_VALID && $stable(RXDAT_DATA) &&
//      $stable(RXDAT_RESP) && $stable(RXDAT_TXNID)) until_with RXDAT_READY
//   4. !RXDAT_VALID until (RXRSP_VALID && RXRSP_READY)

diagram chi_readshared_datreturn {
  clock posedge clk;
  disable iff !rst_n;
  ticks 20;

  param REQ_READY_MAX;
  param RSP_LAT_MAX;
  param DAT_START_MAX;
  param DAT_READY_MAX;
  param BURST_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;
  param ID_WIDTH;

  // REQ channel
  lane TXREQ_VALID:  bit      = 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TXREQ_READY:  bit      = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TXREQ_OPCODE: bus[8]   = x x ReadS ReadS x x x x x x x x x x x x x x x x;
  lane TXREQ_ADDR:   bus[32]  = x x C400 C400 x x x x x x x x x x x x x x x x;
  lane TXREQ_SRCID:  bus[8]   = x x 11 11 x x x x x x x x x x x x x x x x;
  lane TXREQ_TXNID:  bus[8]   = x x 2A 2A x x x x x x x x x x x x x x x x;

  // RSP channel
  lane RXRSP_VALID:  bit      = 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RXRSP_READY:  bit      = 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RXRSP_OPCODE: bus[8]   = x x x x x x Comp Comp x x x x x x x x x x x x;
  lane RXRSP_TXNID:  bus[8]   = x x x x x x 2A 2A x x x x x x x x x x x x;

  // DAT channel
  lane RXDAT_VALID:  bit      = 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0;
  lane RXDAT_READY:  bit      = 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0;
  lane RXDAT_DATA:   bus[128] = x x x x x x x x x Q0 Q0 Q1 Q2 x x x x x x x;
  lane RXDAT_RESP:   bus[4]   = x x x x x x x x x SC SC SC SC x x x x x x x;
  lane RXDAT_TXNID:  bus[8]   = x x x x x x x x x 2A 2A 2A 2A x x x x x x x;
  lane RXDAT_LAST:   bit      = 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;

  //
  // Anchors — request, response, data return
  //
  anchor req_issue    = rise(TXREQ_VALID);
  anchor req_accept   = high(TXREQ_VALID) and high(TXREQ_READY);
  anchor rsp_rise     = rise(RXRSP_VALID);
  anchor rsp_accept   = high(RXRSP_VALID) and high(RXRSP_READY);
  anchor dat_rise     = rise(RXDAT_VALID);
  anchor dat_first    = high(RXDAT_VALID) and high(RXDAT_READY);
  anchor dat_last     = high(RXDAT_VALID) and high(RXDAT_READY) and high(RXDAT_LAST);

  //
  // Windows — channel latency and burst progress
  //
  window req_wait       = between req_issue and req_accept [0:REQ_READY_MAX];
  window rsp_latency    = between req_accept and rsp_rise [1:RSP_LAT_MAX];
  window rsp_wait       = between rsp_rise and rsp_accept [0:1];
  window dat_start      = between rsp_accept and dat_rise [1:DAT_START_MAX];
  window dat_first_wait = between dat_rise and dat_first [0:DAT_READY_MAX];
  window dat_phase      = between dat_rise and dat_last [0:BURST_MAX];

  //
  // Cuts — hide unrelated traffic
  //
  cut pre_req   = before req_issue omitted label "fabric idle";
  cut post_dat  = after dat_last omitted label "transaction retires";

  //
  // Constraints — hold REQ/RSP/DAT metadata until acceptance
  //
  show high(TXREQ_VALID) from req_issue until req_accept;
  show stable(TXREQ_OPCODE) from req_issue until req_accept;
  show stable(TXREQ_ADDR) from req_issue until req_accept;
  show stable(TXREQ_SRCID) from req_issue until req_accept;
  show stable(TXREQ_TXNID) from req_issue until req_accept;

  show high(RXRSP_VALID) from rsp_rise until rsp_accept;
  show stable(RXRSP_OPCODE) from rsp_rise until rsp_accept;
  show stable(RXRSP_TXNID) from rsp_rise until rsp_accept;

  show high(RXDAT_VALID) from dat_rise until dat_last;
  show stable(RXDAT_DATA) from dat_rise until dat_first;
  show stable(RXDAT_RESP) from dat_rise until dat_last;
  show stable(RXDAT_TXNID) from dat_rise until dat_last;

  show low(RXDAT_VALID) before rsp_accept;
  show high(RXDAT_READY) at dat_last;
}
