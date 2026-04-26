diagram td_backpressure_compact_td_000092 {
  clock posedge clk;
  ticks 9;

  lane d: bit = 0 1 1 1 1 0 0 0 0;
  lane c: bit = 0 0 0 0 1 0 0 0 0;
  lane s: bit = 0 0 0 0 1 0 0 0 0;
  lane tg: bus[8] = x D0 D0 D0 D0 x 01 x x;
  lane da: bus[32] = 10 x D0 x x x x x x;
  lane r: bit = 0 0 0 0 0 0 0 0 0;
  lane a: bus[32] = 55 F0 F0 F0 F0 x x x x;
  lane id: bus[32] = x x x F0 FF x x x x;
  lane bus_0: bus[8] = 01 33 x x x F0 x x x;
  lane g: bit;
  lane x: bit;

  anchor valid_rise = rise(d);
  anchor handshake = high(c) and high(s);

  window wait_window = between valid_rise and handshake [2:4];

  show stable(tg) from valid_rise until handshake;
  show neq(da, F0) before valid_rise;
  show high(d) from valid_rise until handshake;
  show low(r) from valid_rise until handshake;
  show stable(a) from valid_rise until handshake;
  show id = FF at handshake;
  show change(bus_0) in wait_window;
}
