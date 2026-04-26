diagram td_backpressure_memory_td_000088 {
  clock posedge mclk;
  ticks 10;

  lane rsp_valid: bit = 0 0 1 1 1 1 0 0 0 0;
  lane rsp_ready: bit = 0 0 0 0 0 1 0 0 0 0;
  lane cmd_ready: bit = 0 0 0 0 0 1 0 0 0 0;
  lane addr: bus[8] = x x 0A 0A 0A 0A x x x 1F;
  lane cmd_valid: bit = 0 0 0 0 0 0 0 0 0 0;
  lane wdata: bus[16] = x 55 01 01 01 01 x x x x;
  lane bit_0: bit = 0 0 0 0 0 1 0 0 0 0;
  lane rdata: bus[16] = x x 02 02 02 02 x x FF x;
  lane bus_0: bus[32] = x x x x 00 10 10 10 10 10;

  anchor valid_rise = rise(rsp_valid);
  anchor handshake = high(rsp_ready) and high(cmd_ready);

  window wait_window = between valid_rise and handshake [2:3];

  show stable(addr) from valid_rise until handshake;
  show low(rsp_ready) before valid_rise;
  show high(rsp_valid) from valid_rise until handshake;
  show low(cmd_valid) from valid_rise until handshake;
  show stable(wdata) from valid_rise until handshake;
  show high(bit_0) at handshake;
  show rdata = 02 in wait_window;
  show stable(bus_0) after handshake;
}
