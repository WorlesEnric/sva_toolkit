property ar_wait(int AR_READY_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(ARVALID) |-> ##[0:AR_READY_MAX] (ARVALID && ARREADY);
endproperty

property r_start(int R_START_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    (ARVALID && ARREADY) |-> ##[1:R_START_MAX] $rose(RVALID);
endproperty

property r_first_wait(int R_READY_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(RVALID) |-> ##[0:R_READY_MAX] (RVALID && RREADY);
endproperty

property r_burst_phase(int BURST_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(RVALID) |-> ##[0:BURST_MAX] (RVALID && RREADY && RLAST);
endproperty

property r_complete(int BURST_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    (RVALID && RREADY) |-> ##[0:BURST_MAX] (RVALID && RREADY && RLAST);
endproperty

property high_ARVALID_from_ar_valid_rise_until_ar_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(ARVALID) |-> ARVALID until_with (ARVALID && ARREADY);
endproperty

property stable_ARADDR_from_ar_valid_rise_until_ar_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(ARVALID) |-> $stable(ARADDR) until_with (ARVALID && ARREADY);
endproperty

property stable_ARLEN_from_ar_valid_rise_until_ar_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(ARVALID) |-> $stable(ARLEN) until_with (ARVALID && ARREADY);
endproperty

property stable_ARSIZE_from_ar_valid_rise_until_ar_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(ARVALID) |-> $stable(ARSIZE) until_with (ARVALID && ARREADY);
endproperty

property stable_ARBURST_from_ar_valid_rise_until_ar_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(ARVALID) |-> $stable(ARBURST) until_with (ARVALID && ARREADY);
endproperty

property stable_ARID_from_ar_valid_rise_until_ar_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(ARVALID) |-> $stable(ARID) until_with (ARVALID && ARREADY);
endproperty

property high_RVALID_from_r_valid_rise_until_r_last_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(RVALID) |-> RVALID until_with (RVALID && RREADY && RLAST);
endproperty

property stable_RDATA_from_r_valid_rise_until_r_first_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(RVALID) |-> $stable(RDATA) until_with (RVALID && RREADY);
endproperty

property stable_RRESP_from_r_valid_rise_until_r_last_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(RVALID) |-> $stable(RRESP) until_with (RVALID && RREADY && RLAST);
endproperty

property stable_RID_from_r_valid_rise_until_r_last_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(RVALID) |-> $stable(RID) until_with (RVALID && RREADY && RLAST);
endproperty

property low_RVALID_before_ar_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    !RVALID until (ARVALID && ARREADY);
endproperty

property high_RREADY_at_r_last_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    (RVALID && RREADY && RLAST) |-> RREADY;
endproperty
