property desc_wait(int DESC_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(DESC_VALID) |-> ##[0:DESC_READY_MAX] (DESC_VALID && DESC_READY);
endproperty

property src_acquire(int SRC_GNT_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(SRC_REQ) |-> ##[0:SRC_GNT_MAX] (SRC_REQ && SRC_GNT);
endproperty

property data_start;
  @(posedge clk) disable iff (!rst_n)
    (SRC_REQ && SRC_GNT) |-> ##[0:2] $rose(BEAT_VALID);
endproperty

property beat_first_wait(int BEAT_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(BEAT_VALID) |-> ##[0:BEAT_READY_MAX] (BEAT_VALID && BEAT_READY);
endproperty

property data_phase;
  @(posedge clk) disable iff (!rst_n)
    $rose(BEAT_VALID) |-> ##[0:256] (BEAT_VALID && BEAT_READY && LAST);
endproperty

property cpl_latency(int CMPL_MAX);
  @(posedge clk) disable iff (!rst_n)
    (BEAT_VALID && BEAT_READY && LAST) |-> ##[1:CMPL_MAX] $rose(CMPL_VALID);
endproperty

property cpl_wait;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMPL_VALID) |-> ##[0:4] (CMPL_VALID && CMPL_READY);
endproperty

property irq_latency;
  @(posedge clk) disable iff (!rst_n)
    (CMPL_VALID && CMPL_READY) |-> ##[0:1] $rose(IRQ);
endproperty

property irq_hold(int IRQ_ACK_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(IRQ) |-> ##[0:IRQ_ACK_MAX] (IRQ && IRQ_ACK);
endproperty

property high_DESC_VALID_from_desc_issue_until_desc_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(DESC_VALID) |-> DESC_VALID until_with (DESC_VALID && DESC_READY);
endproperty

property stable_DESC_ADDR_from_desc_issue_until_desc_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(DESC_VALID) |-> $stable(DESC_ADDR) until_with (DESC_VALID && DESC_READY);
endproperty

property stable_DESC_LEN_from_desc_issue_until_desc_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(DESC_VALID) |-> $stable(DESC_LEN) until_with (DESC_VALID && DESC_READY);
endproperty

property stable_DESC_TAG_from_desc_issue_until_desc_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(DESC_VALID) |-> $stable(DESC_TAG) until_with (DESC_VALID && DESC_READY);
endproperty

property high_SRC_REQ_from_src_req_rise_until_src_grant;
  @(posedge clk) disable iff (!rst_n)
    $rose(SRC_REQ) |-> SRC_REQ until_with (SRC_REQ && SRC_GNT);
endproperty

property low_BEAT_VALID_before_src_grant;
  @(posedge clk) disable iff (!rst_n)
    !BEAT_VALID until (SRC_REQ && SRC_GNT);
endproperty

property high_BEAT_VALID_from_beat_rise_until_beat_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(BEAT_VALID) |-> BEAT_VALID until_with (BEAT_VALID && BEAT_READY && LAST);
endproperty

property stable_DATA_from_beat_rise_until_beat_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(BEAT_VALID) |-> $stable(DATA) until_with (BEAT_VALID && BEAT_READY);
endproperty

property stable_BYTE_EN_from_beat_rise_until_beat_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(BEAT_VALID) |-> $stable(BYTE_EN) until_with (BEAT_VALID && BEAT_READY);
endproperty

property high_CMPL_VALID_from_cpl_rise_until_cpl_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMPL_VALID) |-> CMPL_VALID until_with (CMPL_VALID && CMPL_READY);
endproperty

property stable_STATUS_from_cpl_rise_until_cpl_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMPL_VALID) |-> $stable(STATUS) until_with (CMPL_VALID && CMPL_READY);
endproperty

property stable_CMPL_TAG_from_cpl_rise_until_cpl_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMPL_VALID) |-> $stable(CMPL_TAG) until_with (CMPL_VALID && CMPL_READY);
endproperty

property high_IRQ_from_irq_rise_until_irq_ack;
  @(posedge clk) disable iff (!rst_n)
    $rose(IRQ) |-> IRQ until_with (IRQ && IRQ_ACK);
endproperty

property high_IRQ_ACK_at_irq_ack;
  @(posedge clk) disable iff (!rst_n)
    (IRQ && IRQ_ACK) |-> IRQ_ACK;
endproperty
