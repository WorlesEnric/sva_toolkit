diagram td_burst_compact_td_000046 {
  clock posedge clk;
  ticks 14;

  param M;
  param N;

  lane da: bus[8] = 01 01 01 01 A1 x x x x x x x x x;
  lane d: bit = 0 1 1 1 0 0 0 0 0 0 0 0 0 0;
  lane v: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane b: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane x: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane c: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane r: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane id: bus[32] = x 33 33 33 x x x x x x x x 1F x;
  lane a: bus[8] = x 00 00 00 x x x x x x 55 x x x;
  lane tg: bus[16] = 1F EE FF 0A 00 x x x x x x x 10 x;

  anchor first_beat = high(d) and high(v);
  anchor middle_beat = high(b) and high(x);
  anchor last_beat = high(c) and high(r) and eq(da, A1);

  window burst_first_to_middle = between first_beat and middle_beat [1:M];
  window burst_window = between middle_beat and last_beat [1:N];

  show high(d) from first_beat until middle_beat;
  show stable(id) from first_beat until middle_beat;
  show stable(a) from first_beat until middle_beat;
  show change(tg) at first_beat;
  show change(tg) at middle_beat;
  show change(tg) at last_beat;
}
