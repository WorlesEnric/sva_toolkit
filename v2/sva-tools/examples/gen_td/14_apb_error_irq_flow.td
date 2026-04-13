diagram access_wait__bundle_18 {
  clock posedge PCLK;
  disable iff !PRESETn;

  param WAIT_MAX;
  param IRQ_ACK_MAX;
  param ERR_LATENCY_MAX;

  lane PENABLE: bit;
  lane PREADY: bit;
  lane PSEL: bit;
  lane PSLVERR: bit;
  lane ERR_IRQ: bit;
  lane IRQ_ACK: bit;
  lane PWRITE: bit;
  lane ERR_CODE: bus;
  lane PADDR: bus;
  lane PPROT: bus;
  lane PSTRB: bus;
  lane PWDATA: bus;

  anchor access_wait__bundle_18__node_0 = rise(PENABLE);
  anchor access_wait__bundle_18__point_1 = 1'b1;
  anchor access_wait__bundle_18__node_2 = 1'b1;
  anchor access_wait__bundle_18__node_3 = high(PSEL) and high(PENABLE) and high(PREADY) and high(PSLVERR);
  anchor access_wait__bundle_18__node_4 = rise(ERR_IRQ);
  anchor access_wait__bundle_18__node_5 = high(ERR_IRQ) and high(IRQ_ACK);
  anchor access_wait__bundle_18__node_6 = high(ERR_IRQ) and high(IRQ_ACK);
  anchor access_wait__bundle_18__node_7 = high(IRQ_ACK);
  anchor access_wait__bundle_18__node_8 = rise(PSEL);
  anchor access_wait__bundle_18__node_9 = high(PSEL) and high(PENABLE) and high(PREADY) and high(PSLVERR);
  anchor access_wait__bundle_18__node_10 = rise(ERR_IRQ);
  anchor access_wait__bundle_18__point_11 = 1'b1;
  anchor access_wait__bundle_18__node_12 = rise(PENABLE);

  window access_wait__bundle_18__win_0 = between access_wait__bundle_18__node_0 and access_wait__bundle_18__point_1 [0:0];
  window access_wait__bundle_18__win_1 = between access_wait__bundle_18__point_1 and access_wait__bundle_18__node_2 [0:0];
  window access_wait__bundle_18__win_2 = between access_wait__bundle_18__node_2 and access_wait__bundle_18__node_3 [0:WAIT_MAX];
  window access_wait__bundle_18__win_3 = between access_wait__bundle_18__node_4 and access_wait__bundle_18__point_1 [0:0];
  window access_wait__bundle_18__win_4 = between access_wait__bundle_18__point_1 and access_wait__bundle_18__node_5 [0:0];
  window access_wait__bundle_18__win_5 = between access_wait__bundle_18__node_6 and access_wait__bundle_18__point_1 [0:0];
  window access_wait__bundle_18__win_6 = between access_wait__bundle_18__point_1 and access_wait__bundle_18__node_7 [0:0];
  window access_wait__bundle_18__win_7 = between access_wait__bundle_18__point_1 and access_wait__bundle_18__node_3 [0:0];
  window access_wait__bundle_18__win_8 = between access_wait__bundle_18__node_8 and access_wait__bundle_18__point_1 [0:0];
  window access_wait__bundle_18__win_9 = between access_wait__bundle_18__node_2 and access_wait__bundle_18__node_5 [0:IRQ_ACK_MAX];
  window access_wait__bundle_18__win_10 = between access_wait__bundle_18__node_9 and access_wait__bundle_18__point_1 [0:0];
  window access_wait__bundle_18__win_11 = between access_wait__bundle_18__node_2 and access_wait__bundle_18__node_10 [0:ERR_LATENCY_MAX];
  window access_wait__bundle_18__win_12 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__point_1 [0:0];
  window access_wait__bundle_18__win_13 = between access_wait__bundle_18__node_2 and access_wait__bundle_18__node_12 [1:1];
  window access_wait__bundle_18__win_14 = between access_wait__bundle_18__node_2 and access_wait__bundle_18__node_3 [1:WAIT_MAX];
  window access_wait__bundle_18__win_15 = between access_wait__bundle_18__node_3 and access_wait__bundle_18__node_0 [0:0];
  window access_wait__bundle_18__win_16 = between access_wait__bundle_18__node_3 and access_wait__bundle_18__node_8 [0:0];
  window access_wait__bundle_18__win_17 = between access_wait__bundle_18__node_3 and access_wait__bundle_18__node_9 [0:0];
  window access_wait__bundle_18__win_18 = between access_wait__bundle_18__node_5 and access_wait__bundle_18__node_6 [0:0];
  window access_wait__bundle_18__win_19 = between access_wait__bundle_18__node_5 and access_wait__bundle_18__node_4 [0:0];
  window access_wait__bundle_18__win_20 = between access_wait__bundle_18__node_10 and access_wait__bundle_18__node_4 [0:0];
  window high_ERR_IRQ_from_irq_rise_until_irq_ack__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_6 [0:$];
  window high_PENABLE_from_access_start_until_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window high_PSEL_from_setup_start_until_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window high_PWRITE_from_setup_start_until_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window low_ERR_IRQ_before_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window low_PREADY_before_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window low_PSLVERR_before_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window stable_ERR_CODE_from_irq_rise_until_irq_ack__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_6 [0:$];
  window stable_PADDR_from_setup_start_until_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window stable_PPROT_from_setup_start_until_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window stable_PSTRB_from_setup_start_until_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];
  window stable_PWDATA_from_setup_start_until_error_rsp__hold_2 = between access_wait__bundle_18__point_11 and access_wait__bundle_18__node_9 [0:$];

  show high(ERR_IRQ) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_6;
  show high(PENABLE) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show high(PSEL) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show high(PWRITE) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show low(ERR_IRQ) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show low(PREADY) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show low(PSLVERR) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show stable(ERR_CODE) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_6;
  show stable(PADDR) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show stable(PPROT) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show stable(PSTRB) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;
  show stable(PWDATA) from access_wait__bundle_18__point_11 until access_wait__bundle_18__node_9;

  property access_wait:
    $rose(PENABLE) |-> ##[0:WAIT_MAX] PSEL && PENABLE && PREADY && PSLVERR;
  property high_ERR_IRQ_from_irq_rise_until_irq_ack:
    $rose(ERR_IRQ) |-> ERR_IRQ until_with ERR_IRQ && IRQ_ACK;
  property high_IRQ_ACK_at_irq_ack:
    ERR_IRQ && IRQ_ACK |-> IRQ_ACK;
  property high_PENABLE_from_access_start_until_error_rsp:
    $rose(PENABLE) |-> PENABLE until_with PSEL && PENABLE && PREADY && PSLVERR;
  property high_PSEL_from_setup_start_until_error_rsp:
    $rose(PSEL) |-> PSEL until_with PSEL && PENABLE && PREADY && PSLVERR;
  property high_PWRITE_from_setup_start_until_error_rsp:
    $rose(PSEL) |-> PWRITE until_with PSEL && PENABLE && PREADY && PSLVERR;
  property irq_hold:
    $rose(ERR_IRQ) |-> ##[0:IRQ_ACK_MAX] ERR_IRQ && IRQ_ACK;
  property irq_latency:
    PSEL && PENABLE && PREADY && PSLVERR |-> ##[0:ERR_LATENCY_MAX] $rose(ERR_IRQ);
  property low_ERR_IRQ_before_error_rsp [lossy]:
    !ERR_IRQ until PSEL && PENABLE && PREADY && PSLVERR;
  // low_ERR_IRQ_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_ERR_IRQ_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_ERR_IRQ_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  property low_PREADY_before_error_rsp [lossy]:
    !PREADY until PSEL && PENABLE && PREADY && PSLVERR;
  // low_PREADY_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_PREADY_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_PREADY_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  property low_PSLVERR_before_error_rsp [lossy]:
    !PSLVERR until PSEL && PENABLE && PREADY && PSLVERR;
  // low_PSLVERR_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_PSLVERR_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_PSLVERR_before_error_rsp: until excludes the terminating cycle and is shown as a hold-until skeleton
  property setup_phase:
    $rose(PSEL) |-> ##1 $rose(PENABLE);
  property stable_ERR_CODE_from_irq_rise_until_irq_ack:
    $rose(ERR_IRQ) |-> $stable(ERR_CODE) until_with ERR_IRQ && IRQ_ACK;
  property stable_PADDR_from_setup_start_until_error_rsp:
    $rose(PSEL) |-> $stable(PADDR) until_with PSEL && PENABLE && PREADY && PSLVERR;
  property stable_PPROT_from_setup_start_until_error_rsp:
    $rose(PSEL) |-> $stable(PPROT) until_with PSEL && PENABLE && PREADY && PSLVERR;
  property stable_PSTRB_from_setup_start_until_error_rsp:
    $rose(PSEL) |-> $stable(PSTRB) until_with PSEL && PENABLE && PREADY && PSLVERR;
  property stable_PWDATA_from_setup_start_until_error_rsp:
    $rose(PSEL) |-> $stable(PWDATA) until_with PSEL && PENABLE && PREADY && PSLVERR;
  property write_total:
    $rose(PSEL) |-> ##[1:WAIT_MAX] PSEL && PENABLE && PREADY && PSLVERR;
}