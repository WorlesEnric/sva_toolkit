property aw_latency(int AW_READY_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(AWVALID) |-> ##[0:AW_READY_MAX] (AWVALID && AWREADY);
endproperty

property w_start;
  @(posedge ACLK) disable iff (!ARESETn)
    (AWVALID && AWREADY) |-> ##[0:4] $rose(WVALID);
endproperty

property w_data_phase;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(WVALID) |-> ##[0:128] (WVALID && WREADY && WLAST);
endproperty

property w_beat_ready(int W_READY_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(WVALID) |-> ##[0:W_READY_MAX] (WVALID && WREADY);
endproperty

property b_latency(int BRESP_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    (WVALID && WREADY && WLAST) |-> ##[1:BRESP_MAX] $rose(BVALID);
endproperty

property b_handshake_w;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(BVALID) |-> ##[0:16] (BVALID && BREADY);
endproperty

property high_AWVALID_from_aw_valid_rise_until_aw_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(AWVALID) |-> AWVALID until_with (AWVALID && AWREADY);
endproperty

property stable_AWADDR_from_aw_valid_rise_until_aw_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(AWVALID) |-> $stable(AWADDR) until_with (AWVALID && AWREADY);
endproperty

property stable_AWLEN_from_aw_valid_rise_until_aw_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(AWVALID) |-> $stable(AWLEN) until_with (AWVALID && AWREADY);
endproperty

property stable_AWSIZE_from_aw_valid_rise_until_aw_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(AWVALID) |-> $stable(AWSIZE) until_with (AWVALID && AWREADY);
endproperty

property stable_AWBURST_from_aw_valid_rise_until_aw_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(AWVALID) |-> $stable(AWBURST) until_with (AWVALID && AWREADY);
endproperty

property high_WVALID_from_w_first_beat_until_w_last_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(WVALID) |-> WVALID until_with (WVALID && WREADY && WLAST);
endproperty

property stable_WDATA_from_w_first_beat_until_w_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(WVALID) |-> $stable(WDATA) until_with (WVALID && WREADY);
endproperty

property stable_WSTRB_from_w_first_beat_until_w_beat;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(WVALID) |-> $stable(WSTRB) until_with (WVALID && WREADY);
endproperty

property high_BVALID_from_b_valid_rise_until_b_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(BVALID) |-> BVALID until_with (BVALID && BREADY);
endproperty

property stable_BRESP_from_b_valid_rise_until_b_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    $rose(BVALID) |-> $stable(BRESP) until_with (BVALID && BREADY);
endproperty

property low_WVALID_before_aw_handshake;
  @(posedge ACLK) disable iff (!ARESETn)
    !WVALID until (AWVALID && AWREADY);
endproperty
