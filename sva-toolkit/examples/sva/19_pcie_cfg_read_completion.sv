property req_wait(int REQ_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(CFG_REQ_VALID) |-> ##[0:REQ_READY_MAX] (CFG_REQ_VALID && CFG_REQ_READY);
endproperty

property cpl_start(int CPL_START_MAX);
  @(posedge clk) disable iff (!rst_n)
    (CFG_REQ_VALID && CFG_REQ_READY) |-> ##[1:CPL_START_MAX] $rose(CPL_VALID);
endproperty

property cpl_first_wait(int CPL_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(CPL_VALID) |-> ##[0:CPL_READY_MAX] (CPL_VALID && CPL_READY);
endproperty

property cpl_phase;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPL_VALID) |-> ##[0:8] (CPL_VALID && CPL_READY && CPL_LAST);
endproperty

property fc_latency(int FC_UPDATE_MAX);
  @(posedge clk) disable iff (!rst_n)
    (CPL_VALID && CPL_READY && CPL_LAST) |-> ##[0:FC_UPDATE_MAX] $rose(FC_UPDATE_VALID);
endproperty

property fc_wait;
  @(posedge clk) disable iff (!rst_n)
    $rose(FC_UPDATE_VALID) |-> ##[0:1] (FC_UPDATE_VALID && FC_UPDATE_READY);
endproperty

property high_CFG_REQ_VALID_from_cfg_req_rise_until_cfg_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CFG_REQ_VALID) |-> CFG_REQ_VALID until_with (CFG_REQ_VALID && CFG_REQ_READY);
endproperty

property stable_CFG_FMT_TYPE_from_cfg_req_rise_until_cfg_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CFG_REQ_VALID) |-> $stable(CFG_FMT_TYPE) until_with (CFG_REQ_VALID && CFG_REQ_READY);
endproperty

property stable_CFG_ADDR_from_cfg_req_rise_until_cfg_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CFG_REQ_VALID) |-> $stable(CFG_ADDR) until_with (CFG_REQ_VALID && CFG_REQ_READY);
endproperty

property stable_CFG_TAG_from_cfg_req_rise_until_cfg_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CFG_REQ_VALID) |-> $stable(CFG_TAG) until_with (CFG_REQ_VALID && CFG_REQ_READY);
endproperty

property stable_CFG_BDF_from_cfg_req_rise_until_cfg_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CFG_REQ_VALID) |-> $stable(CFG_BDF) until_with (CFG_REQ_VALID && CFG_REQ_READY);
endproperty

property high_CPL_VALID_from_cpl_rise_until_cpl_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPL_VALID) |-> CPL_VALID until_with (CPL_VALID && CPL_READY && CPL_LAST);
endproperty

property stable_CPL_STATUS_from_cpl_rise_until_cpl_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPL_VALID) |-> $stable(CPL_STATUS) until_with (CPL_VALID && CPL_READY && CPL_LAST);
endproperty

property stable_CPL_DATA_from_cpl_rise_until_cpl_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPL_VALID) |-> $stable(CPL_DATA) until_with (CPL_VALID && CPL_READY);
endproperty

property stable_CPL_TAG_from_cpl_rise_until_cpl_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPL_VALID) |-> $stable(CPL_TAG) until_with (CPL_VALID && CPL_READY && CPL_LAST);
endproperty

property high_FC_UPDATE_VALID_from_fc_rise_until_fc_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(FC_UPDATE_VALID) |-> FC_UPDATE_VALID until_with (FC_UPDATE_VALID && FC_UPDATE_READY);
endproperty

property stable_FC_HDR_CRED_from_fc_rise_until_fc_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(FC_UPDATE_VALID) |-> $stable(FC_HDR_CRED) until_with (FC_UPDATE_VALID && FC_UPDATE_READY);
endproperty

property stable_FC_DATA_CRED_from_fc_rise_until_fc_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(FC_UPDATE_VALID) |-> $stable(FC_DATA_CRED) until_with (FC_UPDATE_VALID && FC_UPDATE_READY);
endproperty

property low_CPL_VALID_before_cfg_req_hs;
  @(posedge clk) disable iff (!rst_n)
    !CPL_VALID until (CFG_REQ_VALID && CFG_REQ_READY);
endproperty

property low_FC_UPDATE_VALID_before_cpl_last;
  @(posedge clk) disable iff (!rst_n)
    !FC_UPDATE_VALID until (CPL_VALID && CPL_READY && CPL_LAST);
endproperty
