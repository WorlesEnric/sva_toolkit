diagram cpl_first_wait__bundle_20 {
  clock posedge clk;
  disable iff !rst_n;

  param CPL_READY_MAX;
  param CPL_START_MAX;
  param FC_UPDATE_MAX;
  param REQ_READY_MAX;

  lane CPL_READY: bit;
  lane CPL_VALID: bit;
  lane CPL_LAST: bit;
  lane CFG_REQ_READY: bit;
  lane CFG_REQ_VALID: bit;
  lane FC_UPDATE_VALID: bit;
  lane FC_UPDATE_READY: bit;
  lane CFG_ADDR: bus;
  lane CFG_BDF: bus;
  lane CFG_FMT_TYPE: bus;
  lane CFG_TAG: bus;
  lane CPL_DATA: bus;
  lane CPL_STATUS: bus;
  lane CPL_TAG: bus;
  lane FC_DATA_CRED: bus;
  lane FC_HDR_CRED: bus;

  anchor cpl_first_wait__bundle_20__node_0 = rise(CPL_VALID);
  anchor cpl_first_wait__bundle_20__point_1 = 1'b1;
  anchor cpl_first_wait__bundle_20__node_2 = 1'b1;
  anchor cpl_first_wait__bundle_20__node_3 = high(CPL_VALID) and high(CPL_READY);
  anchor cpl_first_wait__bundle_20__node_4 = high(CPL_VALID) and high(CPL_READY) and high(CPL_LAST);
  anchor cpl_first_wait__bundle_20__node_5 = high(CFG_REQ_VALID) and high(CFG_REQ_READY);
  anchor cpl_first_wait__bundle_20__node_6 = rise(CPL_VALID);
  anchor cpl_first_wait__bundle_20__node_7 = high(CPL_VALID) and high(CPL_READY) and high(CPL_LAST);
  anchor cpl_first_wait__bundle_20__node_8 = rise(FC_UPDATE_VALID);
  anchor cpl_first_wait__bundle_20__node_9 = rise(FC_UPDATE_VALID);
  anchor cpl_first_wait__bundle_20__node_10 = high(FC_UPDATE_VALID) and high(FC_UPDATE_READY);
  anchor cpl_first_wait__bundle_20__node_11 = rise(CFG_REQ_VALID);
  anchor cpl_first_wait__bundle_20__node_12 = high(CFG_REQ_VALID) and high(CFG_REQ_READY);
  anchor cpl_first_wait__bundle_20__point_13 = 1'b1;

  window cpl_first_wait__bundle_20__win_0 = between cpl_first_wait__bundle_20__node_0 and cpl_first_wait__bundle_20__point_1 [0:0];
  window cpl_first_wait__bundle_20__win_1 = between cpl_first_wait__bundle_20__point_1 and cpl_first_wait__bundle_20__node_2 [0:0];
  window cpl_first_wait__bundle_20__win_2 = between cpl_first_wait__bundle_20__node_2 and cpl_first_wait__bundle_20__node_3 [0:CPL_READY_MAX];
  window cpl_first_wait__bundle_20__win_3 = between cpl_first_wait__bundle_20__node_2 and cpl_first_wait__bundle_20__node_4 [0:8];
  window cpl_first_wait__bundle_20__win_4 = between cpl_first_wait__bundle_20__node_5 and cpl_first_wait__bundle_20__point_1 [0:0];
  window cpl_first_wait__bundle_20__win_5 = between cpl_first_wait__bundle_20__node_2 and cpl_first_wait__bundle_20__node_6 [1:CPL_START_MAX];
  window cpl_first_wait__bundle_20__win_6 = between cpl_first_wait__bundle_20__node_7 and cpl_first_wait__bundle_20__point_1 [0:0];
  window cpl_first_wait__bundle_20__win_7 = between cpl_first_wait__bundle_20__node_2 and cpl_first_wait__bundle_20__node_8 [0:FC_UPDATE_MAX];
  window cpl_first_wait__bundle_20__win_8 = between cpl_first_wait__bundle_20__node_9 and cpl_first_wait__bundle_20__point_1 [0:0];
  window cpl_first_wait__bundle_20__win_9 = between cpl_first_wait__bundle_20__node_2 and cpl_first_wait__bundle_20__node_10 [0:1];
  window cpl_first_wait__bundle_20__win_10 = between cpl_first_wait__bundle_20__node_11 and cpl_first_wait__bundle_20__point_1 [0:0];
  window cpl_first_wait__bundle_20__win_11 = between cpl_first_wait__bundle_20__point_1 and cpl_first_wait__bundle_20__node_12 [0:0];
  window cpl_first_wait__bundle_20__win_12 = between cpl_first_wait__bundle_20__point_1 and cpl_first_wait__bundle_20__node_4 [0:0];
  window cpl_first_wait__bundle_20__win_13 = between cpl_first_wait__bundle_20__point_1 and cpl_first_wait__bundle_20__node_10 [0:0];
  window cpl_first_wait__bundle_20__win_14 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__point_1 [0:0];
  window cpl_first_wait__bundle_20__win_15 = between cpl_first_wait__bundle_20__node_2 and cpl_first_wait__bundle_20__node_12 [0:REQ_READY_MAX];
  window cpl_first_wait__bundle_20__win_16 = between cpl_first_wait__bundle_20__point_1 and cpl_first_wait__bundle_20__node_3 [0:0];
  window cpl_first_wait__bundle_20__win_17 = between cpl_first_wait__bundle_20__node_3 and cpl_first_wait__bundle_20__node_0 [0:0];
  window cpl_first_wait__bundle_20__win_18 = between cpl_first_wait__bundle_20__node_3 and cpl_first_wait__bundle_20__node_7 [0:0];
  window cpl_first_wait__bundle_20__win_19 = between cpl_first_wait__bundle_20__node_4 and cpl_first_wait__bundle_20__node_7 [0:0];
  window cpl_first_wait__bundle_20__win_20 = between cpl_first_wait__bundle_20__node_4 and cpl_first_wait__bundle_20__node_0 [0:0];
  window cpl_first_wait__bundle_20__win_21 = between cpl_first_wait__bundle_20__node_6 and cpl_first_wait__bundle_20__node_7 [0:0];
  window cpl_first_wait__bundle_20__win_22 = between cpl_first_wait__bundle_20__node_6 and cpl_first_wait__bundle_20__node_0 [0:0];
  window cpl_first_wait__bundle_20__win_23 = between cpl_first_wait__bundle_20__node_8 and cpl_first_wait__bundle_20__node_9 [0:0];
  window cpl_first_wait__bundle_20__win_24 = between cpl_first_wait__bundle_20__node_10 and cpl_first_wait__bundle_20__node_9 [0:0];
  window cpl_first_wait__bundle_20__win_25 = between cpl_first_wait__bundle_20__node_12 and cpl_first_wait__bundle_20__node_11 [0:0];
  window high_CFG_REQ_VALID_from_cfg_req_rise_until_cfg_req_hs__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_5 [0:$];
  window high_CPL_VALID_from_cpl_rise_until_cpl_last__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_7 [0:$];
  window low_CPL_VALID_before_cfg_req_hs__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_5 [0:$];
  window low_FC_UPDATE_VALID_before_cpl_last__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_7 [0:$];
  window stable_CFG_ADDR_from_cfg_req_rise_until_cfg_req_hs__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_5 [0:$];
  window stable_CFG_BDF_from_cfg_req_rise_until_cfg_req_hs__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_5 [0:$];
  window stable_CFG_FMT_TYPE_from_cfg_req_rise_until_cfg_req_hs__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_5 [0:$];
  window stable_CFG_TAG_from_cfg_req_rise_until_cfg_req_hs__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_5 [0:$];
  window stable_CPL_STATUS_from_cpl_rise_until_cpl_last__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_7 [0:$];
  window stable_CPL_TAG_from_cpl_rise_until_cpl_last__hold_2 = between cpl_first_wait__bundle_20__point_13 and cpl_first_wait__bundle_20__node_7 [0:$];

  show high(CFG_REQ_VALID) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_5;
  show high(CPL_VALID) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_7;
  show low(CPL_VALID) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_5;
  show low(FC_UPDATE_VALID) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_7;
  show stable(CFG_ADDR) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_5;
  show stable(CFG_BDF) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_5;
  show stable(CFG_FMT_TYPE) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_5;
  show stable(CFG_TAG) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_5;
  show stable(CPL_STATUS) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_7;
  show stable(CPL_TAG) from cpl_first_wait__bundle_20__point_13 until cpl_first_wait__bundle_20__node_7;

  property cpl_first_wait:
    $rose(CPL_VALID) |-> ##[0:CPL_READY_MAX] CPL_VALID && CPL_READY;
  property cpl_phase:
    $rose(CPL_VALID) |-> ##[0:8] CPL_VALID && CPL_READY && CPL_LAST;
  property cpl_start:
    CFG_REQ_VALID && CFG_REQ_READY |-> ##[1:CPL_START_MAX] $rose(CPL_VALID);
  property fc_latency:
    CPL_VALID && CPL_READY && CPL_LAST |-> ##[0:FC_UPDATE_MAX] $rose(FC_UPDATE_VALID);
  property fc_wait:
    $rose(FC_UPDATE_VALID) |-> ##[0:1] FC_UPDATE_VALID && FC_UPDATE_READY;
  property high_CFG_REQ_VALID_from_cfg_req_rise_until_cfg_req_hs:
    $rose(CFG_REQ_VALID) |-> CFG_REQ_VALID until_with CFG_REQ_VALID && CFG_REQ_READY;
  property high_CPL_VALID_from_cpl_rise_until_cpl_last:
    $rose(CPL_VALID) |-> CPL_VALID until_with CPL_VALID && CPL_READY && CPL_LAST;
  property high_FC_UPDATE_VALID_from_fc_rise_until_fc_hs:
    $rose(FC_UPDATE_VALID) |-> FC_UPDATE_VALID until_with FC_UPDATE_VALID && FC_UPDATE_READY;
  property low_CPL_VALID_before_cfg_req_hs [lossy]:
    !CPL_VALID until CFG_REQ_VALID && CFG_REQ_READY;
  // low_CPL_VALID_before_cfg_req_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_CPL_VALID_before_cfg_req_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_CPL_VALID_before_cfg_req_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  property low_FC_UPDATE_VALID_before_cpl_last [lossy]:
    !FC_UPDATE_VALID until CPL_VALID && CPL_READY && CPL_LAST;
  // low_FC_UPDATE_VALID_before_cpl_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_FC_UPDATE_VALID_before_cpl_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_FC_UPDATE_VALID_before_cpl_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  property req_wait:
    $rose(CFG_REQ_VALID) |-> ##[0:REQ_READY_MAX] CFG_REQ_VALID && CFG_REQ_READY;
  property stable_CFG_ADDR_from_cfg_req_rise_until_cfg_req_hs:
    $rose(CFG_REQ_VALID) |-> $stable(CFG_ADDR) until_with CFG_REQ_VALID && CFG_REQ_READY;
  property stable_CFG_BDF_from_cfg_req_rise_until_cfg_req_hs:
    $rose(CFG_REQ_VALID) |-> $stable(CFG_BDF) until_with CFG_REQ_VALID && CFG_REQ_READY;
  property stable_CFG_FMT_TYPE_from_cfg_req_rise_until_cfg_req_hs:
    $rose(CFG_REQ_VALID) |-> $stable(CFG_FMT_TYPE) until_with CFG_REQ_VALID && CFG_REQ_READY;
  property stable_CFG_TAG_from_cfg_req_rise_until_cfg_req_hs:
    $rose(CFG_REQ_VALID) |-> $stable(CFG_TAG) until_with CFG_REQ_VALID && CFG_REQ_READY;
  property stable_CPL_DATA_from_cpl_rise_until_cpl_first:
    $rose(CPL_VALID) |-> $stable(CPL_DATA) until_with CPL_VALID && CPL_READY;
  property stable_CPL_STATUS_from_cpl_rise_until_cpl_last:
    $rose(CPL_VALID) |-> $stable(CPL_STATUS) until_with CPL_VALID && CPL_READY && CPL_LAST;
  property stable_CPL_TAG_from_cpl_rise_until_cpl_last:
    $rose(CPL_VALID) |-> $stable(CPL_TAG) until_with CPL_VALID && CPL_READY && CPL_LAST;
  property stable_FC_DATA_CRED_from_fc_rise_until_fc_hs:
    $rose(FC_UPDATE_VALID) |-> $stable(FC_DATA_CRED) until_with FC_UPDATE_VALID && FC_UPDATE_READY;
  property stable_FC_HDR_CRED_from_fc_rise_until_fc_hs:
    $rose(FC_UPDATE_VALID) |-> $stable(FC_HDR_CRED) until_with FC_UPDATE_VALID && FC_UPDATE_READY;
}