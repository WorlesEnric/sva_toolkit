// Example 14: APB write with wait states, error completion, and IRQ reporting
// Scenario:
//   A peripheral write stalls in the APB access phase, completes with PSLVERR,
//   then the bridge raises an interrupt and waits for firmware acknowledgement.
// Reference properties:
//   1. $rose(PSEL) |-> (PSEL && PWRITE && $stable(PADDR) && $stable(PWDATA) &&
//      $stable(PSTRB) && $stable(PPROT)) until_with (PSEL && PENABLE && PREADY && PSLVERR)
//   2. $rose(PENABLE) |-> PENABLE until_with (PSEL && PENABLE && PREADY && PSLVERR)
//   3. $rose(ERR_IRQ) |-> (ERR_IRQ && $stable(ERR_CODE)) until_with IRQ_ACK
//   4. !ERR_IRQ until (PSEL && PENABLE && PREADY && PSLVERR)

diagram apb_error_irq_flow {
  clock posedge PCLK;
  disable iff !PRESETn;
  ticks 16;

  param WAIT_MAX;
  param IRQ_ACK_MAX;
  param ERR_LATENCY_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;

  lane PSEL:     bit     = 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0;
  lane PENABLE:  bit     = 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0;
  lane PWRITE:   bit     = 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0;
  lane PREADY:   bit     = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane PSLVERR:  bit     = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane PADDR:    bus[32] = x 0040 0040 0040 0040 0040 x x x x x x x x x x;
  lane PWDATA:   bus[32] = x DEAD DEAD DEAD DEAD DEAD x x x x x x x x x x;
  lane PSTRB:    bus[4]  = x 0F 0F 0F 0F 0F x x x x x x x x x x;
  lane PPROT:    bus[3]  = x 000 000 000 000 000 x x x x x x x x x x;
  lane ERR_IRQ:  bit     = 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0;
  lane IRQ_ACK:  bit     = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane ERR_CODE: bus[4]  = x x x x x x 5 5 5 x x x x x x x;

  //
  // Anchors — APB transaction phases
  //
  anchor setup_start  = rise(PSEL);
  anchor access_start = rise(PENABLE);
  anchor error_rsp    = high(PSEL) and high(PENABLE) and high(PREADY) and high(PSLVERR);

  //
  // Anchors — software-visible reporting
  //
  anchor irq_rise     = rise(ERR_IRQ);
  anchor irq_ack      = high(ERR_IRQ) and high(IRQ_ACK);

  //
  // Windows — setup, wait-state access, and irq reporting
  //
  window setup_phase   = between setup_start and access_start [1:1];
  window access_wait   = between access_start and error_rsp [0:WAIT_MAX];
  window irq_latency   = between error_rsp and irq_rise [0:ERR_LATENCY_MAX];
  window irq_hold      = between irq_rise and irq_ack [0:IRQ_ACK_MAX];
  window write_total   = between setup_start and error_rsp [1:WAIT_MAX];

  //
  // Cuts — focus on the failed transfer and software response
  //
  cut pre_setup   = before setup_start omitted label "bus idle";
  cut post_irq    = after irq_ack omitted label "handler returns";

  //
  // Constraints — APB control must remain stable through the error completion
  //
  show high(PSEL) from setup_start until error_rsp;
  show high(PWRITE) from setup_start until error_rsp;
  show high(PENABLE) from access_start until error_rsp;
  show stable(PADDR) from setup_start until error_rsp;
  show stable(PWDATA) from setup_start until error_rsp;
  show stable(PSTRB) from setup_start until error_rsp;
  show stable(PPROT) from setup_start until error_rsp;

  show low(PSLVERR) before error_rsp;
  show low(PREADY) before error_rsp;

  show high(ERR_IRQ) from irq_rise until irq_ack;
  show stable(ERR_CODE) from irq_rise until irq_ack;
  show low(ERR_IRQ) before error_rsp;
  show high(IRQ_ACK) at irq_ack;
}
