// Example 16: DMA descriptor acceptance through completion and interrupt
// Scenario:
//   A DMA engine accepts a descriptor, waits for source grant, streams a burst
//   of write data with first-beat backpressure, returns a completion status,
//   and raises an interrupt until software acknowledgement.
// Reference properties:
//   1. (DESC_VALID && !DESC_READY) |-> (DESC_VALID && $stable(DESC_ADDR) &&
//      $stable(DESC_LEN) && $stable(DESC_TAG)) until_with DESC_READY
//   2. (SRC_REQ && !SRC_GNT) |-> SRC_REQ until_with SRC_GNT
//   3. (BEAT_VALID && !BEAT_READY) |-> (BEAT_VALID && $stable(DATA) && $stable(BYTE_EN)) until_with BEAT_READY
//   4. $rose(CMPL_VALID) |-> (CMPL_VALID && $stable(STATUS) && $stable(CMPL_TAG)) until_with CMPL_READY
//   5. $rose(IRQ) |-> IRQ until_with IRQ_ACK

diagram dma_descriptor_completion {
  clock posedge clk;
  disable iff !rst_n;
  ticks 20;

  param DESC_READY_MAX;
  param SRC_GNT_MAX;
  param BEAT_READY_MAX;
  param CMPL_MAX;
  param IRQ_ACK_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;

  // Descriptor input
  lane DESC_VALID: bit     = 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane DESC_READY: bit     = 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane DESC_ADDR:  bus[32] = x x 9000 x x x x x x x x x x x x x x x x x;
  lane DESC_LEN:   bus[16] = x x 0004 x x x x x x x x x x x x x x x x x;
  lane DESC_TAG:   bus[8]  = x x 3A x x x x x x x x x x x x x x x x x;

  // Resource acquisition
  lane SRC_REQ:    bit     = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane SRC_GNT:    bit     = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  // Burst data phase
  lane BEAT_VALID: bit     = 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane BEAT_READY: bit     = 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane DATA:       bus[64] = x x x x x W0 W0 W1 W2 x x x x x x x x x x x;
  lane BYTE_EN:    bus[8]  = x x x x x FF FF FF FF x x x x x x x x x x x;
  lane LAST:       bit     = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;

  // Completion and software notification
  lane CMPL_VALID: bit     = 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0;
  lane CMPL_READY: bit     = 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane STATUS:     bus[4]  = x x x x x x x x x x OK OK x x x x x x x x;
  lane CMPL_TAG:   bus[8]  = x x x x x x x x x x 3A 3A x x x x x x x x;
  lane IRQ:        bit     = 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0;
  lane IRQ_ACK:    bit     = 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;

  //
  // Anchors — descriptor, data, completion, interrupt
  //
  anchor desc_issue    = rise(DESC_VALID);
  anchor desc_accept   = high(DESC_VALID) and high(DESC_READY);
  anchor src_req_rise  = rise(SRC_REQ);
  anchor src_grant     = high(SRC_REQ) and high(SRC_GNT);
  anchor beat_rise     = rise(BEAT_VALID);
  anchor beat_first    = high(BEAT_VALID) and high(BEAT_READY);
  anchor beat_last     = high(BEAT_VALID) and high(BEAT_READY) and high(LAST);
  anchor cpl_rise      = rise(CMPL_VALID);
  anchor cpl_handshake = high(CMPL_VALID) and high(CMPL_READY);
  anchor irq_rise      = rise(IRQ);
  anchor irq_ack       = high(IRQ) and high(IRQ_ACK);

  //
  // Windows — end-to-end DMA execution
  //
  window desc_wait        = between desc_issue and desc_accept [0:DESC_READY_MAX];
  window src_acquire      = between src_req_rise and src_grant [0:SRC_GNT_MAX];
  window data_start       = between src_grant and beat_rise [0:2];
  window beat_first_wait  = between beat_rise and beat_first [0:BEAT_READY_MAX];
  window data_phase       = between beat_rise and beat_last [0:256];
  window cpl_latency      = between beat_last and cpl_rise [1:CMPL_MAX];
  window cpl_wait         = between cpl_rise and cpl_handshake [0:4];
  window irq_latency      = between cpl_handshake and irq_rise [0:1];
  window irq_hold         = between irq_rise and irq_ack [0:IRQ_ACK_MAX];

  //
  // Cuts — compress reset/idle time outside the active DMA span
  //
  cut pre_desc   = before desc_issue omitted label "queue idle";
  cut post_irq   = after irq_ack omitted label "driver re-arms channel";

  //
  // Constraints — each phase must hold its payload until acceptance
  //
  show high(DESC_VALID) from desc_issue until desc_accept;
  show stable(DESC_ADDR) from desc_issue until desc_accept;
  show stable(DESC_LEN) from desc_issue until desc_accept;
  show stable(DESC_TAG) from desc_issue until desc_accept;

  show high(SRC_REQ) from src_req_rise until src_grant;
  show low(BEAT_VALID) before src_grant;

  show high(BEAT_VALID) from beat_rise until beat_last;
  show stable(DATA) from beat_rise until beat_first;
  show stable(BYTE_EN) from beat_rise until beat_first;

  show high(CMPL_VALID) from cpl_rise until cpl_handshake;
  show stable(STATUS) from cpl_rise until cpl_handshake;
  show stable(CMPL_TAG) from cpl_rise until cpl_handshake;

  show high(IRQ) from irq_rise until irq_ack;
  show high(IRQ_ACK) at irq_ack;
}
