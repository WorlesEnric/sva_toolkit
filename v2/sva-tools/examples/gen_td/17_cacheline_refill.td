diagram high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21 {
  clock posedge clk;
  disable iff !rst_n;

  param MEM_DATA_MAX;
  param MEM_REQ_MAX;
  param MISS_GNT_MAX;
  param RESP_READY_MAX;

  lane CPU_RESP_READY: bit;
  lane CPU_RESP_VALID: bit;
  lane MEM_ARREADY: bit;
  lane MEM_ARVALID: bit;
  lane MEM_RLAST: bit;
  lane MEM_RREADY: bit;
  lane MEM_RVALID: bit;
  lane MISS_READY: bit;
  lane MISS_VALID: bit;
  lane MSHR_BUSY: bit;
  lane REFILL_BUSY: bit;
  lane MEM_ARADDR: bus;
  lane MEM_ARLEN: bus;
  lane MEM_RDATA: bus;
  lane MISS_ADDR: bus;
  lane MISS_ID: bus;
  lane RESP_ID: bus;
  lane RESP_STATUS: bus;

  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_0 = rise(CPU_RESP_VALID);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 = 1'b1;
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_2 = high(CPU_RESP_VALID) and high(CPU_RESP_READY);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_3 = rise(MEM_ARVALID);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_4 = high(MEM_ARVALID) and high(MEM_ARREADY);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_5 = rise(MEM_RVALID);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_6 = high(MEM_RVALID) and high(MEM_RREADY) and high(MEM_RLAST);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_7 = rise(MISS_VALID);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_8 = high(MISS_VALID) and high(MISS_READY);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9 = high(MISS_VALID) and high(MISS_READY);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 = 1'b1;
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 = 1'b1;
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_12 = high(MEM_RVALID) and high(MEM_RREADY);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13 = high(MEM_RVALID) and high(MEM_RREADY) and high(MEM_RLAST);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_14 = rise(CPU_RESP_VALID);
  anchor high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_15 = rise(MEM_ARVALID);

  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_0 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_0 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_1 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_2 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_2 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_3 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_3 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_4 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_4 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_5 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_5 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_6 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_6 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_7 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_7 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_8 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_8 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_9 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_10 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_11 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_6 [0:MEM_DATA_MAX];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_12 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_12 [0:2];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_13 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_4 [0:MEM_REQ_MAX];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_14 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_8 [0:MISS_GNT_MAX];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_15 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_16 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_14 [0:2];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_17 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_15 [0:1];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_18 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_11 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_2 [0:RESP_READY_MAX];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_19 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_1 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_12 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_20 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_2 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_0 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_21 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_4 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_3 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_22 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_6 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_5 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_23 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_6 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_24 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_8 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_25 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_8 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_7 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_26 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_12 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_27 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_12 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_5 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_28 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_14 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_0 [0:0];
  window high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__win_29 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_15 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_3 [0:0];
  window high_MEM_RVALID_from_mem_r_rise_until_mem_last__hold_2 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13 [0:$];
  window high_MISS_VALID_from_miss_issue_until_miss_accept__hold_2 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9 [0:$];
  window low_CPU_RESP_VALID_before_mem_last__hold_2 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13 [0:$];
  window stable_MISS_ADDR_from_miss_issue_until_miss_accept__hold_2 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9 [0:$];
  window stable_MISS_ID_from_miss_issue_until_miss_accept__hold_2 = between high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 and high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9 [0:$];

  show high(MEM_RVALID) from high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 until high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13;
  show high(MISS_VALID) from high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 until high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9;
  show low(CPU_RESP_VALID) from high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 until high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_13;
  show stable(MISS_ADDR) from high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 until high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9;
  show stable(MISS_ID) from high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__point_10 until high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs__bundle_21__node_9;

  property high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs:
    $rose(CPU_RESP_VALID) |-> CPU_RESP_VALID until_with CPU_RESP_VALID && CPU_RESP_READY;
  property high_MEM_ARVALID_from_mem_req_rise_until_mem_req_hs:
    $rose(MEM_ARVALID) |-> MEM_ARVALID until_with MEM_ARVALID && MEM_ARREADY;
  property high_MEM_RVALID_from_mem_r_rise_until_mem_last:
    $rose(MEM_RVALID) |-> MEM_RVALID until_with MEM_RVALID && MEM_RREADY && MEM_RLAST;
  property high_MISS_VALID_from_miss_issue_until_miss_accept:
    $rose(MISS_VALID) |-> MISS_VALID until_with MISS_VALID && MISS_READY;
  property high_MSHR_BUSY_from_miss_accept_until_cpu_rsp_hs:
    MISS_VALID && MISS_READY |-> MSHR_BUSY until_with CPU_RESP_VALID && CPU_RESP_READY;
  property high_REFILL_BUSY_from_mem_r_rise_until_cpu_rsp_hs:
    $rose(MEM_RVALID) |-> REFILL_BUSY until_with CPU_RESP_VALID && CPU_RESP_READY;
  property low_CPU_RESP_VALID_before_mem_last [lossy]:
    !CPU_RESP_VALID until MEM_RVALID && MEM_RREADY && MEM_RLAST;
  // low_CPU_RESP_VALID_before_mem_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_CPU_RESP_VALID_before_mem_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_CPU_RESP_VALID_before_mem_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  property mem_data_phase:
    $rose(MEM_RVALID) |-> ##[0:MEM_DATA_MAX] MEM_RVALID && MEM_RREADY && MEM_RLAST;
  property mem_first_wait:
    $rose(MEM_RVALID) |-> ##[0:2] MEM_RVALID && MEM_RREADY;
  property mem_req_wait:
    $rose(MEM_ARVALID) |-> ##[0:MEM_REQ_MAX] MEM_ARVALID && MEM_ARREADY;
  property miss_wait:
    $rose(MISS_VALID) |-> ##[0:MISS_GNT_MAX] MISS_VALID && MISS_READY;
  property refill_done:
    MEM_RVALID && MEM_RREADY && MEM_RLAST |-> ##[0:2] $rose(CPU_RESP_VALID);
  property req_gap:
    MISS_VALID && MISS_READY |-> ##[0:1] $rose(MEM_ARVALID);
  property resp_wait:
    $rose(CPU_RESP_VALID) |-> ##[0:RESP_READY_MAX] CPU_RESP_VALID && CPU_RESP_READY;
  property stable_MEM_ARADDR_from_mem_req_rise_until_mem_req_hs:
    $rose(MEM_ARVALID) |-> $stable(MEM_ARADDR) until_with MEM_ARVALID && MEM_ARREADY;
  property stable_MEM_ARLEN_from_mem_req_rise_until_mem_req_hs:
    $rose(MEM_ARVALID) |-> $stable(MEM_ARLEN) until_with MEM_ARVALID && MEM_ARREADY;
  property stable_MEM_RDATA_from_mem_r_rise_until_mem_first:
    $rose(MEM_RVALID) |-> $stable(MEM_RDATA) until_with MEM_RVALID && MEM_RREADY;
  property stable_MISS_ADDR_from_miss_issue_until_miss_accept:
    $rose(MISS_VALID) |-> $stable(MISS_ADDR) until_with MISS_VALID && MISS_READY;
  property stable_MISS_ID_from_miss_issue_until_miss_accept:
    $rose(MISS_VALID) |-> $stable(MISS_ID) until_with MISS_VALID && MISS_READY;
  property stable_RESP_ID_from_cpu_rsp_rise_until_cpu_rsp_hs:
    $rose(CPU_RESP_VALID) |-> $stable(RESP_ID) until_with CPU_RESP_VALID && CPU_RESP_READY;
  property stable_RESP_STATUS_from_cpu_rsp_rise_until_cpu_rsp_hs:
    $rose(CPU_RESP_VALID) |-> $stable(RESP_STATUS) until_with CPU_RESP_VALID && CPU_RESP_READY;
}