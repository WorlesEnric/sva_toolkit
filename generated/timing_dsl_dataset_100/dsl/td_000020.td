diagram td_backpressure_memory_td_000020 {
  clock negedge mclk;
  ticks 10;

  lane cmd_valid: bit = 0 0 1 1 1 0 0 0 0 0;
  lane rsp_valid: bit = 0 0 0 0 1 0 0 0 0 0;
  lane rsp_ready: bit = 0 0 0 0 1 0 0 0 0 0;
  lane wdata: bus[8] = x x 55 55 55 x x D0 x x;
  lane addr: bus[16] = x x A5 A5 A5 x A1 x x x;
  lane cmd_ready: bit = 0 0 0 0 0 0 0 0 0 0;
  lane bit_0: bit = 0 1 0 0 0 0 0 0 0 0;
  lane rdata: bus[32] = x x D0 D0 D0 x x x 7E x;
  lane bus_0: bus[16] = x 10 F0 x x A5 x x x x;

  anchor valid_rise = rise(cmd_valid);
  anchor handshake = high(rsp_valid) and high(rsp_ready);

  window wait_window = between valid_rise and handshake [2:4];

  show wdata = 55 from valid_rise until handshake;
  show stable(addr) in wait_window;
  show low(cmd_ready) before handshake;
  show high(cmd_valid) from valid_rise until handshake;
  show low(bit_0) from valid_rise until handshake;
  show stable(rdata) from valid_rise until handshake;
  show change(bus_0) in wait_window;
}
