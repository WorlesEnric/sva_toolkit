diagram td_backpressure_memory_td_000060 {
  clock posedge mclk;
  ticks 13;

  lane cmd_ready: bit = 0 1 1 1 1 0 0 0 0 0 0 0 0;
  lane cmd_valid: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane rsp_valid: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane rsp_ready: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane addr: bus[32] = F0 0A 0A 0A 0A x x x x x x x x;
  lane bit_0: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0;

  anchor valid_rise = rise(cmd_ready);
  anchor handshake = high(cmd_valid) and high(rsp_valid);

  window wait_window = between valid_rise and handshake [3:3];

  show low(cmd_valid) before valid_rise;
  show high(cmd_ready) from valid_rise until handshake;
  show low(rsp_ready) from valid_rise until handshake;
  show stable(addr) from valid_rise until handshake;
  show fall(bit_0) at handshake;
}
