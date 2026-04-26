diagram td_backpressure_generic_td_000075 {
  clock posedge clk;
  ticks 11;

  param MAX_LAT;

  lane req: bit = 0 1 1 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 1 0 0 0 0 0 0 0 0;
  lane stop: bit = 0 0 1 0 0 0 0 0 0 0 0;
  lane addr: bus[8] = x 55 55 x x x A5 x x x x;
  lane ready: bit = 0 0 0 0 0 1 0 0 0 0 0;
  lane tag: bus[32] = x FF FF A5 x x x x x x x;
  lane data: bus[8] = x EE EE x x x x x 1F x x;
  lane busy: bit = 0 0 0 0 0 0 0 0 0 0 0;

  anchor valid_rise = rise(req);
  anchor handshake = high(ack) and high(stop);

  window wait_window = between valid_rise and handshake [1:MAX_LAT];

  show stable(addr) in wait_window;
  show high(req) from valid_rise until handshake;
  show low(ready) from valid_rise until handshake;
  show stable(tag) from valid_rise until handshake;
  show data = EE in wait_window;
}
