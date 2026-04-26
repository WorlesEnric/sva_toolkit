diagram td_backpressure_memory_td_000093 {
  clock posedge mclk;
  ticks 18;

  lane rsp_valid: bit = 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane cmd_valid: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane cmd_ready: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane wdata: bus[8] = x 00 00 00 00 x x x x x FF x x x x x x x;
  lane addr: bus[16] = x A1 A1 A1 A1 x x x x x x x x x x x x 02;
  lane rsp_ready: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane rdata: bus[8] = x BE BE BE BE x x x x x x A1 x x x x x x;
  lane bus_0: bus[32] = x EE EE EE EE x x x x x x x x F0 x x x x;
  lane bus_1: bus[8] = x EE x x 0A 0A 0A 0A 0A 0A 0A 0A 0A 0A 0A 0A 0A 0A;
  lane bit_0: bit;

  anchor valid_rise = rise(rsp_valid);
  anchor handshake = high(cmd_valid) and high(cmd_ready);

  window wait_window = between valid_rise and handshake [1:4];

  cut cut_before_93 = before valid_rise omitted label "idle";

  show high(rsp_valid) from valid_rise until handshake;
  show stable(wdata) from valid_rise until handshake;
  show stable(addr) from valid_rise until handshake;
  show low(cmd_valid) before valid_rise;
  show high(rsp_valid) from valid_rise until handshake;
  show low(rsp_ready) from valid_rise until handshake;
  show stable(rdata) from valid_rise until handshake;
  show stable(bus_0) in wait_window;
  show bus_1 = 0A after handshake;
}
