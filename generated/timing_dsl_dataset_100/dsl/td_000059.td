diagram td_chain_memory_td_000059 {
  clock posedge mclk;
  ticks 7;

  lane addr: bus[16];
  lane rdata: bus[16];
  lane cmd_valid: bit;
  lane rsp_ready: bit;
  lane cmd_ready: bit;
  lane rsp_valid: bit;
  lane bit_0: bit;
  lane wdata: bus[16];

  anchor trigger = rise(cmd_valid);
  anchor middle = high(rsp_ready) and high(cmd_ready) and eq(addr, A1);
  anchor response = change(rdata);

  window first_window = between trigger and middle [1:2];
  window second_window = between middle and response [2:2];

  show stable(rsp_valid) from trigger until middle;
  show high(rsp_ready) from middle until response;
  show low(bit_0) before trigger;
  show neq(wdata, 7E) after response;
}
