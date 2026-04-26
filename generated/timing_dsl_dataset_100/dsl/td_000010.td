diagram td_burst_compact_td_000010 {
  clock posedge clk;
  ticks 7;

  param M;
  param N;

  lane a: bus[16] = F0 F0 F0 F0 BE x x;
  lane s: bit = 0 0 1 1 0 0 0;
  lane d: bit = 0 0 1 0 0 0 0;
  lane g: bit = 0 0 0 1 0 0 0;
  lane r: bit = 0 0 0 1 0 0 0;
  lane c: bit = 0 0 0 0 1 0 0;
  lane x: bit = 0 0 0 0 1 0 0;
  lane da: bus[16] = x A1 10 10 x x x;
  lane id: bus[16] = x 55 A5 A5 x x x;
  lane tg: bus[32] = 55 01 F0 01 1F x x;

  anchor first_beat = high(s) and high(d);
  anchor middle_beat = high(g) and high(r);
  anchor last_beat = high(c) and high(x) and eq(a, BE);

  window burst_first_to_middle = between first_beat and middle_beat [1:M];
  window burst_window = between middle_beat and last_beat [1:N];

  show high(s) from first_beat until middle_beat;
  show stable(da) in burst_first_to_middle;
  show stable(id) in burst_first_to_middle;
  show low(c) before first_beat;
  show change(tg) at first_beat;
  show change(tg) at middle_beat;
  show change(tg) at last_beat;
}
