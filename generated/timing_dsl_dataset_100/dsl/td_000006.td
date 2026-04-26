diagram td_backpressure_memory_td_000006 {
  clock posedge mclk;
  ticks 11;

  lane cmd_ready: bit = 0 1 1 0 0 0 0 0 0 0 0;
  lane cmd_valid: bit = 0 0 1 0 0 0 0 0 0 0 0;
  lane rsp_ready: bit = 0 0 1 0 0 0 0 0 0 0 0;
  lane rdata: bus[16] = x 1F 1F x x x x x A1 x x;
  lane addr: bus[32] = A1 33 33 x x x x x x x x;
  lane rsp_valid: bit = 1 0 0 0 0 0 0 0 0 0 0;
  lane wdata: bus[8] = x D0 D0 x x x 1F x x x x;
  lane bit_0: bit = 0 1 0 0 0 0 0 0 0 0 0;
  lane bus_0: bus[32] = 0A A5 x x x 10 x x x x x;
  lane bit_1: bit = 0 0 1 1 1 1 1 1 1 1 1;

  anchor valid_rise = rise(cmd_ready);
  anchor handshake = high(cmd_valid) and high(rsp_ready);

  window wait_window = between valid_rise and handshake [1:1];

  show high(cmd_ready) from valid_rise until handshake;
  show stable(rdata) from valid_rise until handshake;
  show stable(addr) from valid_rise until handshake;
  show low(cmd_valid) before valid_rise;
  show high(cmd_ready) from valid_rise until handshake;
  show low(rsp_valid) from valid_rise until handshake;
  show stable(wdata) from valid_rise until handshake;
  show fall(bit_0) at handshake;
  show change(bus_0) in wait_window;
  show high(bit_1) after handshake;
}
