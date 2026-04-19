property req_wait(int REQ_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(TXREQ_VALID) |-> ##[0:REQ_READY_MAX] (TXREQ_VALID && TXREQ_READY);
endproperty

property rsp_latency(int RSP_LAT_MAX);
  @(posedge clk) disable iff (!rst_n)
    (TXREQ_VALID && TXREQ_READY) |-> ##[1:RSP_LAT_MAX] $rose(RXRSP_VALID);
endproperty

property rsp_wait;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXRSP_VALID) |-> ##[0:1] (RXRSP_VALID && RXRSP_READY);
endproperty

property dat_start(int DAT_START_MAX);
  @(posedge clk) disable iff (!rst_n)
    (RXRSP_VALID && RXRSP_READY) |-> ##[1:DAT_START_MAX] $rose(RXDAT_VALID);
endproperty

property dat_first_wait(int DAT_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(RXDAT_VALID) |-> ##[0:DAT_READY_MAX] (RXDAT_VALID && RXDAT_READY);
endproperty

property dat_phase(int BURST_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(RXDAT_VALID) |-> ##[0:BURST_MAX] (RXDAT_VALID && RXDAT_READY && RXDAT_LAST);
endproperty

property high_TXREQ_VALID_from_req_issue_until_req_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(TXREQ_VALID) |-> TXREQ_VALID until_with (TXREQ_VALID && TXREQ_READY);
endproperty

property stable_TXREQ_OPCODE_from_req_issue_until_req_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(TXREQ_VALID) |-> $stable(TXREQ_OPCODE) until_with (TXREQ_VALID && TXREQ_READY);
endproperty

property stable_TXREQ_ADDR_from_req_issue_until_req_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(TXREQ_VALID) |-> $stable(TXREQ_ADDR) until_with (TXREQ_VALID && TXREQ_READY);
endproperty

property stable_TXREQ_SRCID_from_req_issue_until_req_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(TXREQ_VALID) |-> $stable(TXREQ_SRCID) until_with (TXREQ_VALID && TXREQ_READY);
endproperty

property stable_TXREQ_TXNID_from_req_issue_until_req_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(TXREQ_VALID) |-> $stable(TXREQ_TXNID) until_with (TXREQ_VALID && TXREQ_READY);
endproperty

property high_RXRSP_VALID_from_rsp_rise_until_rsp_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXRSP_VALID) |-> RXRSP_VALID until_with (RXRSP_VALID && RXRSP_READY);
endproperty

property stable_RXRSP_OPCODE_from_rsp_rise_until_rsp_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXRSP_VALID) |-> $stable(RXRSP_OPCODE) until_with (RXRSP_VALID && RXRSP_READY);
endproperty

property stable_RXRSP_TXNID_from_rsp_rise_until_rsp_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXRSP_VALID) |-> $stable(RXRSP_TXNID) until_with (RXRSP_VALID && RXRSP_READY);
endproperty

property high_RXDAT_VALID_from_dat_rise_until_dat_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXDAT_VALID) |-> RXDAT_VALID until_with (RXDAT_VALID && RXDAT_READY && RXDAT_LAST);
endproperty

property stable_RXDAT_DATA_from_dat_rise_until_dat_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXDAT_VALID) |-> $stable(RXDAT_DATA) until_with (RXDAT_VALID && RXDAT_READY);
endproperty

property stable_RXDAT_RESP_from_dat_rise_until_dat_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXDAT_VALID) |-> $stable(RXDAT_RESP) until_with (RXDAT_VALID && RXDAT_READY && RXDAT_LAST);
endproperty

property stable_RXDAT_TXNID_from_dat_rise_until_dat_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(RXDAT_VALID) |-> $stable(RXDAT_TXNID) until_with (RXDAT_VALID && RXDAT_READY && RXDAT_LAST);
endproperty

property low_RXDAT_VALID_before_rsp_accept;
  @(posedge clk) disable iff (!rst_n)
    !RXDAT_VALID until (RXRSP_VALID && RXRSP_READY);
endproperty

property high_RXDAT_READY_at_dat_last;
  @(posedge clk) disable iff (!rst_n)
    (RXDAT_VALID && RXDAT_READY && RXDAT_LAST) |-> RXDAT_READY;
endproperty
