diagram td_burst_compact_td_000070 {
  clock posedge clk;
  ticks 14;

  lane a: bus[8] = FF FF FF FF FF FF FF FF FF FF 1F x x x;
  lane g: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane b: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane c: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane x: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane r: bit = 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
  lane d: bit = 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
  lane v: bit = 0 0 1 1 1 1 1 0 0 0 0 0 0 0;
  lane tg: bus[16] = 10 x D0 D0 D0 D0 D0 x x x x x x x;
  lane id: bus[32] = x x BE BE BE BE BE x x x 0A x x x;
  lane da: bus[8] = x A1 33 x x 0A 55 x x F0 D0 x 01 x;
  lane bus_0: bus[8] = x x x x x x x x 10 x BE BE BE BE;

  anchor first_beat = high(g) and high(b);
  anchor middle_beat = high(c) and high(x);
  anchor last_beat = high(r) and high(d) and eq(a, 1F);

  window burst_first_to_middle = between first_beat and middle_beat [2:4];
  window burst_window = between middle_beat and last_beat [2:4];

  show stable(v) from first_beat until middle_beat;
  show stable(tg) from first_beat until middle_beat;
  show stable(id) from first_beat until middle_beat;
  show low(r) before first_beat;
  show change(da) at first_beat;
  show change(da) at middle_beat;
  show change(da) at last_beat;
  show bus_0 = BE after last_beat;
}
