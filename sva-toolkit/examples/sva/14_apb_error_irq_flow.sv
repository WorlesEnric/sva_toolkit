property setup_phase;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> ##1 $rose(PENABLE);
endproperty

property access_wait(int WAIT_MAX);
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PENABLE) |-> ##[0:WAIT_MAX] (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property irq_latency(int ERR_LATENCY_MAX);
  @(posedge PCLK) disable iff (!PRESETn)
    (PSEL && PENABLE && PREADY && PSLVERR) |-> ##[0:ERR_LATENCY_MAX] $rose(ERR_IRQ);
endproperty

property irq_hold(int IRQ_ACK_MAX);
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(ERR_IRQ) |-> ##[0:IRQ_ACK_MAX] (ERR_IRQ && IRQ_ACK);
endproperty

property write_total(int WAIT_MAX);
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> ##[1:WAIT_MAX] (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property high_PSEL_from_setup_start_until_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> PSEL until_with (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property high_PWRITE_from_setup_start_until_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> PWRITE until_with (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property high_PENABLE_from_access_start_until_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PENABLE) |-> PENABLE until_with (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property stable_PADDR_from_setup_start_until_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> $stable(PADDR) until_with (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property stable_PWDATA_from_setup_start_until_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> $stable(PWDATA) until_with (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property stable_PSTRB_from_setup_start_until_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> $stable(PSTRB) until_with (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property stable_PPROT_from_setup_start_until_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(PSEL) |-> $stable(PPROT) until_with (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property low_PSLVERR_before_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    !PSLVERR until (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property low_PREADY_before_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    !PREADY until (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property high_ERR_IRQ_from_irq_rise_until_irq_ack;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(ERR_IRQ) |-> ERR_IRQ until_with (ERR_IRQ && IRQ_ACK);
endproperty

property stable_ERR_CODE_from_irq_rise_until_irq_ack;
  @(posedge PCLK) disable iff (!PRESETn)
    $rose(ERR_IRQ) |-> $stable(ERR_CODE) until_with (ERR_IRQ && IRQ_ACK);
endproperty

property low_ERR_IRQ_before_error_rsp;
  @(posedge PCLK) disable iff (!PRESETn)
    !ERR_IRQ until (PSEL && PENABLE && PREADY && PSLVERR);
endproperty

property high_IRQ_ACK_at_irq_ack;
  @(posedge PCLK) disable iff (!PRESETn)
    (ERR_IRQ && IRQ_ACK) |-> IRQ_ACK;
endproperty
