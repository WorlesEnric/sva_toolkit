diagram td_backpressure_compact_td_000079 {
  clock negedge clk;
  ticks 6;

  lane x: bit = 0 0 1 1 1 0;
  lane s: bit = 0 0 0 0 1 0;
  lane r: bit = 0 0 0 0 1 0;
  lane c: bit = 0 1 0 0 0 0;
  lane id: bus[32] = F0 x 7E 7E 7E x;
  lane v: bit = 0 0 0 0 0 0;
  lane g: bit = 1 0 0 0 0 0;
  lane da: bus[16] = x x 55 55 55 10;
  lane a: bus[8] = x F0 33 33 EE x;
  lane d: bit = 0 0 0 0 1 1;

  anchor valid_rise = rise(x);
  anchor handshake = high(s) and high(r);

  window wait_window = between valid_rise and handshake [2:2];

  show stable(c) from valid_rise until handshake;
  show stable(id) from valid_rise until handshake;
  show low(v) before handshake;
  show high(x) from valid_rise until handshake;
  show low(g) from valid_rise until handshake;
  show stable(da) from valid_rise until handshake;
  show neq(a, 0A) in wait_window;
  show high(d) after handshake;
}
