// Example 17: Cacheline miss refill from miss request to CPU response
// Scenario:
//   A cache miss request is held until acceptance, the refill engine launches a
//   memory read, the first memory beat is backpressured, refill stays busy until
//   the line completes, and the CPU response is held until the core is ready.
// Reference properties:
//   1. (MISS_VALID && !MISS_READY) |-> (MISS_VALID && $stable(MISS_ADDR) && $stable(MISS_ID)) until_with MISS_READY
//   2. (MEM_ARVALID && !MEM_ARREADY) |-> (MEM_ARVALID && $stable(MEM_ARADDR) && $stable(MEM_ARLEN)) until_with MEM_ARREADY
//   3. (MEM_RVALID && !MEM_RREADY) |-> (MEM_RVALID && $stable(MEM_RDATA)) until_with MEM_RREADY
//   4. $rose(CPU_RESP_VALID) |-> (CPU_RESP_VALID && $stable(RESP_ID) && $stable(RESP_STATUS)) until_with CPU_RESP_READY
//   5. !$rose(CPU_RESP_VALID) until (MEM_RVALID && MEM_RREADY && MEM_RLAST)

diagram cacheline_refill {
  clock posedge clk;
  disable iff !rst_n;
  ticks 20;

  param MISS_GNT_MAX;
  param MEM_REQ_MAX;
  param MEM_DATA_MAX;
  param RESP_READY_MAX;
  param ADDR_WIDTH;
  param DATA_WIDTH;
  param ID_WIDTH;

  // Miss request / MSHR allocation
  lane MISS_VALID:    bit     = 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane MISS_READY:    bit     = 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane MISS_ADDR:     bus[32] = x x C000 x x x x x x x x x x x x x x x x x;
  lane MISS_ID:       bus[4]  = x x 3 x x x x x x x x x x x x x x x x x;
  lane MSHR_BUSY:     bit     = 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0;

  // Memory request
  lane MEM_ARVALID:   bit     = 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane MEM_ARREADY:   bit     = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane MEM_ARADDR:    bus[32] = x x x C000 C000 x x x x x x x x x x x x x x x;
  lane MEM_ARLEN:     bus[8]  = x x x 03 03 x x x x x x x x x x x x x x x;

  // Memory data / refill
  lane MEM_RVALID:    bit     = 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane MEM_RREADY:    bit     = 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane MEM_RDATA:     bus[128] = x x x x x C0 C0 C1 C2 x x x x x x x x x x x;
  lane MEM_RLAST:     bit     = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane REFILL_BUSY:   bit     = 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0;

  // CPU response
  lane CPU_RESP_VALID: bit     = 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0;
  lane CPU_RESP_READY: bit     = 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane RESP_ID:        bus[4]  = x x x x x x x x x 3 3 x x x x x x x x x;
  lane RESP_STATUS:    bus[4]  = x x x x x x x x x HIT HIT x x x x x x x x x;

  //
  // Anchors — miss, memory request, refill completion, CPU response
  //
  anchor miss_issue    = rise(MISS_VALID);
  anchor miss_accept   = high(MISS_VALID) and high(MISS_READY);
  anchor mem_req_rise  = rise(MEM_ARVALID);
  anchor mem_req_hs    = high(MEM_ARVALID) and high(MEM_ARREADY);
  anchor mem_r_rise    = rise(MEM_RVALID);
  anchor mem_first     = high(MEM_RVALID) and high(MEM_RREADY);
  anchor mem_last      = high(MEM_RVALID) and high(MEM_RREADY) and high(MEM_RLAST);
  anchor cpu_rsp_rise  = rise(CPU_RESP_VALID);
  anchor cpu_rsp_hs    = high(CPU_RESP_VALID) and high(CPU_RESP_READY);

  //
  // Windows — refill pipeline stages
  //
  window miss_wait      = between miss_issue and miss_accept [0:MISS_GNT_MAX];
  window req_gap        = between miss_accept and mem_req_rise [0:1];
  window mem_req_wait   = between mem_req_rise and mem_req_hs [0:MEM_REQ_MAX];
  window mem_first_wait = between mem_r_rise and mem_first [0:2];
  window mem_data_phase = between mem_r_rise and mem_last [0:MEM_DATA_MAX];
  window refill_done    = between mem_last and cpu_rsp_rise [0:2];
  window resp_wait      = between cpu_rsp_rise and cpu_rsp_hs [0:RESP_READY_MAX];

  //
  // Cuts — focus on the miss / refill / wakeup episode
  //
  cut pre_miss   = before miss_issue omitted label "cache hit traffic elided";
  cut post_resp  = after cpu_rsp_hs omitted label "pipeline resumes";

  //
  // Constraints — payloads and status must remain stable until accepted
  //
  show high(MISS_VALID) from miss_issue until miss_accept;
  show stable(MISS_ADDR) from miss_issue until miss_accept;
  show stable(MISS_ID) from miss_issue until miss_accept;

  show high(MEM_ARVALID) from mem_req_rise until mem_req_hs;
  show stable(MEM_ARADDR) from mem_req_rise until mem_req_hs;
  show stable(MEM_ARLEN) from mem_req_rise until mem_req_hs;

  show high(MEM_RVALID) from mem_r_rise until mem_last;
  show stable(MEM_RDATA) from mem_r_rise until mem_first;
  show high(REFILL_BUSY) from mem_r_rise until cpu_rsp_hs;
  show high(MSHR_BUSY) from miss_accept until cpu_rsp_hs;

  show low(CPU_RESP_VALID) before mem_last;
  show high(CPU_RESP_VALID) from cpu_rsp_rise until cpu_rsp_hs;
  show stable(RESP_ID) from cpu_rsp_rise until cpu_rsp_hs;
  show stable(RESP_STATUS) from cpu_rsp_rise until cpu_rsp_hs;
}
