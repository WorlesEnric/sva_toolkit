property miss_wait(int MISS_GNT_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(MISS_VALID) |-> ##[0:MISS_GNT_MAX] (MISS_VALID && MISS_READY);
endproperty

property req_gap;
  @(posedge clk) disable iff (!rst_n)
    (MISS_VALID && MISS_READY) |-> ##[0:1] $rose(MEM_ARVALID);
endproperty

property mem_req_wait(int MEM_REQ_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_ARVALID) |-> ##[0:MEM_REQ_MAX] (MEM_ARVALID && MEM_ARREADY);
endproperty

property mem_first_wait;
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_RVALID) |-> ##[0:2] (MEM_RVALID && MEM_RREADY);
endproperty

property mem_data_phase(int MEM_DATA_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_RVALID) |-> ##[0:MEM_DATA_MAX] (MEM_RVALID && MEM_RREADY && MEM_RLAST);
endproperty

property refill_done;
  @(posedge clk) disable iff (!rst_n)
    (MEM_RVALID && MEM_RREADY && MEM_RLAST) |-> ##[0:2] $rose(CPU_RESP_VALID);
endproperty

property resp_wait(int RESP_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(CPU_RESP_VALID) |-> ##[0:RESP_READY_MAX] (CPU_RESP_VALID && CPU_RESP_READY);
endproperty

property high_MISS_VALID_from_miss_issue_until_miss_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(MISS_VALID) |-> MISS_VALID until_with (MISS_VALID && MISS_READY);
endproperty

property stable_MISS_ADDR_from_miss_issue_until_miss_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(MISS_VALID) |-> $stable(MISS_ADDR) until_with (MISS_VALID && MISS_READY);
endproperty

property stable_MISS_ID_from_miss_issue_until_miss_accept;
  @(posedge clk) disable iff (!rst_n)
    $rose(MISS_VALID) |-> $stable(MISS_ID) until_with (MISS_VALID && MISS_READY);
endproperty

property high_MEM_ARVALID_from_mem_req_rise_until_mem_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_ARVALID) |-> MEM_ARVALID until_with (MEM_ARVALID && MEM_ARREADY);
endproperty

property stable_MEM_ARADDR_from_mem_req_rise_until_mem_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_ARVALID) |-> $stable(MEM_ARADDR) until_with (MEM_ARVALID && MEM_ARREADY);
endproperty

property stable_MEM_ARLEN_from_mem_req_rise_until_mem_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_ARVALID) |-> $stable(MEM_ARLEN) until_with (MEM_ARVALID && MEM_ARREADY);
endproperty

property high_MEM_RVALID_from_mem_r_rise_until_mem_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_RVALID) |-> MEM_RVALID until_with (MEM_RVALID && MEM_RREADY && MEM_RLAST);
endproperty

property stable_MEM_RDATA_from_mem_r_rise_until_mem_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_RVALID) |-> $stable(MEM_RDATA) until_with (MEM_RVALID && MEM_RREADY);
endproperty

property high_REFILL_BUSY_from_mem_r_rise_until_cpu_rsp_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(MEM_RVALID) |-> REFILL_BUSY until_with (CPU_RESP_VALID && CPU_RESP_READY);
endproperty

property high_MSHR_BUSY_from_miss_accept_until_cpu_rsp_hs;
  @(posedge clk) disable iff (!rst_n)
    (MISS_VALID && MISS_READY) |-> MSHR_BUSY until_with (CPU_RESP_VALID && CPU_RESP_READY);
endproperty

property low_CPU_RESP_VALID_before_mem_last;
  @(posedge clk) disable iff (!rst_n)
    !CPU_RESP_VALID until (MEM_RVALID && MEM_RREADY && MEM_RLAST);
endproperty

property high_CPU_RESP_VALID_from_cpu_rsp_rise_until_cpu_rsp_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPU_RESP_VALID) |-> CPU_RESP_VALID until_with (CPU_RESP_VALID && CPU_RESP_READY);
endproperty

property stable_RESP_ID_from_cpu_rsp_rise_until_cpu_rsp_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPU_RESP_VALID) |-> $stable(RESP_ID) until_with (CPU_RESP_VALID && CPU_RESP_READY);
endproperty

property stable_RESP_STATUS_from_cpu_rsp_rise_until_cpu_rsp_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CPU_RESP_VALID) |-> $stable(RESP_STATUS) until_with (CPU_RESP_VALID && CPU_RESP_READY);
endproperty
