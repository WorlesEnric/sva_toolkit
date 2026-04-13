// Example 19: PCIe configuration read followed by completion and flow-control update
// Scenario:
//   A configuration read request TLP is issued on the request interface, waits
//   for ready, a completion later returns with initial backpressure, and a
//   separate flow-control update is sent once the completion is retired.
// Reference properties:
//   1. (CFG_REQ_VALID && !CFG_REQ_READY) |-> (CFG_REQ_VALID && $stable(CFG_FMT_TYPE) &&
//      $stable(CFG_ADDR) && $stable(CFG_TAG) && $stable(CFG_BDF)) until_with CFG_REQ_READY
//   2. (CPL_VALID && !CPL_READY) |-> (CPL_VALID && $stable(CPL_STATUS) &&
//      $stable(CPL_DATA) && $stable(CPL_TAG)) until_with CPL_READY
//   3. (FC_UPDATE_VALID && !FC_UPDATE_READY) |-> (FC_UPDATE_VALID &&
//      $stable(FC_HDR_CRED) && $stable(FC_DATA_CRED)) until_with FC_UPDATE_READY
//   4. !CPL_VALID until (CFG_REQ_VALID && CFG_REQ_READY)

diagram pcie_cfg_read_completion {
  clock posedge clk;
  disable iff !rst_n;
  ticks 20;

  param REQ_READY_MAX;
  param CPL_START_MAX;
  param CPL_READY_MAX;
  param FC_UPDATE_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;

  // Configuration request
  lane CFG_REQ_VALID: bit      = 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CFG_REQ_READY: bit      = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CFG_FMT_TYPE:  bus[8]   = x x CFG0 CFG0 x x x x x x x x x x x x x x x x;
  lane CFG_ADDR:      bus[16]  = x x 0010 0010 x x x x x x x x x x x x x x x x;
  lane CFG_TAG:       bus[8]   = x x 19 19 x x x x x x x x x x x x x x x x;
  lane CFG_BDF:       bus[16]  = x x 0001 0001 x x x x x x x x x x x x x x x x;

  // Completion return
  lane CPL_VALID:     bit      = 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0;
  lane CPL_READY:     bit      = 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0;
  lane CPL_STATUS:    bus[4]   = x x x x x x x SC SC SC x x x x x x x x x x;
  lane CPL_DATA:      bus[32]  = x x x x x x x D0 D0 D1 x x x x x x x x x x;
  lane CPL_TAG:       bus[8]   = x x x x x x x 19 19 19 x x x x x x x x x x;
  lane CPL_LAST:      bit      = 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;

  // Flow-control update after completion
  lane FC_UPDATE_VALID: bit    = 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0;
  lane FC_UPDATE_READY: bit    = 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
  lane FC_HDR_CRED:    bus[8]  = x x x x x x x x x x x x 08 08 x x x x x x;
  lane FC_DATA_CRED:   bus[8]  = x x x x x x x x x x x x 10 10 x x x x x x;

  //
  // Anchors — request, completion, credit update
  //
  anchor cfg_req_rise  = rise(CFG_REQ_VALID);
  anchor cfg_req_hs    = high(CFG_REQ_VALID) and high(CFG_REQ_READY);
  anchor cpl_rise      = rise(CPL_VALID);
  anchor cpl_first     = high(CPL_VALID) and high(CPL_READY);
  anchor cpl_last      = high(CPL_VALID) and high(CPL_READY) and high(CPL_LAST);
  anchor fc_rise       = rise(FC_UPDATE_VALID);
  anchor fc_hs         = high(FC_UPDATE_VALID) and high(FC_UPDATE_READY);

  //
  // Windows — request wait, completion latency, credit return
  //
  window req_wait       = between cfg_req_rise and cfg_req_hs [0:REQ_READY_MAX];
  window cpl_start      = between cfg_req_hs and cpl_rise [1:CPL_START_MAX];
  window cpl_first_wait = between cpl_rise and cpl_first [0:CPL_READY_MAX];
  window cpl_phase      = between cpl_rise and cpl_last [0:8];
  window fc_latency     = between cpl_last and fc_rise [0:FC_UPDATE_MAX];
  window fc_wait        = between fc_rise and fc_hs [0:1];

  //
  // Cuts — compress idle PCIe link time
  //
  cut pre_req   = before cfg_req_rise omitted label "idle link";
  cut post_fc   = after fc_hs omitted label "credits restored";

  //
  // Constraints — TLP header and completion payload must hold until ready
  //
  show high(CFG_REQ_VALID) from cfg_req_rise until cfg_req_hs;
  show stable(CFG_FMT_TYPE) from cfg_req_rise until cfg_req_hs;
  show stable(CFG_ADDR) from cfg_req_rise until cfg_req_hs;
  show stable(CFG_TAG) from cfg_req_rise until cfg_req_hs;
  show stable(CFG_BDF) from cfg_req_rise until cfg_req_hs;

  show high(CPL_VALID) from cpl_rise until cpl_last;
  show stable(CPL_STATUS) from cpl_rise until cpl_last;
  show stable(CPL_DATA) from cpl_rise until cpl_first;
  show stable(CPL_TAG) from cpl_rise until cpl_last;

  show high(FC_UPDATE_VALID) from fc_rise until fc_hs;
  show stable(FC_HDR_CRED) from fc_rise until fc_hs;
  show stable(FC_DATA_CRED) from fc_rise until fc_hs;

  show low(CPL_VALID) before cfg_req_hs;
  show low(FC_UPDATE_VALID) before cpl_last;
}
