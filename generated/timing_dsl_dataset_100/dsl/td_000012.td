diagram td_single_response_memory_td_000012 {
  clock posedge mclk;
  ticks 11;

  param CMD_MAX;

  lane cmd_valid: bit;
  lane rsp_ready: bit;
  lane rsp_valid: bit;
  lane rdata: bus[16];
  lane addr: bus[32];
  lane cmd_ready: bit;
  lane bit_0: bit;

  anchor trigger = high(cmd_valid);
  anchor response = high(rsp_ready) and high(rsp_valid);

  window response_window = between trigger and response [1:CMD_MAX];

  show rdata = 10 from trigger until response;
  show stable(addr) in response_window;
  show low(cmd_ready) before response;
  show fall(bit_0) at response;
}
