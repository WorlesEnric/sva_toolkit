diagram td_backpressure_generic_td_000095 {
  clock posedge clk;
  ticks 13;

  lane start: bit = 0 1 1 1 1 0 0 0 0 0 0 0 0;
  lane done: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane resp_valid: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane addr: bus[8] = x 00 00 00 00 x x 55 x x x x x;
  lane ready: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane len: bus[16] = x 1F 1F 1F 1F x x x x x x FF x;
  lane tag: bus[32] = x x x x 55 x x x x x x x 7E;
  lane ack: bit = 0 0 0 0 1 1 1 1 1 1 1 1 1;
  lane last: bit;
  lane req: bit;

  anchor valid_rise = rise(start);
  anchor handshake = high(done) and high(resp_valid);

  window wait_window = between valid_rise and handshake [2:4];

  show stable(addr) from valid_rise until handshake;
  show low(done) before valid_rise;
  show high(start) from valid_rise until handshake;
  show low(ready) from valid_rise until handshake;
  show stable(len) from valid_rise until handshake;
  show tag = 55 at handshake;
  show high(ack) after handshake;
}
