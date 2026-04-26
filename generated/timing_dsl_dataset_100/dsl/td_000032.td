diagram td_join_compact_td_000032 {
  clock negedge clk;
  ticks 19;

  param M;

  lane x: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane r: bit = 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane d: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane v: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane da: bus[16] = x x 01 01 x x x x x x x x x x x BE x x x;
  lane tg: bus[32] = x x A5 A5 x x x x x x x x 0A x x x x x x;

  anchor first = rise(x);
  anchor second = high(r);
  anchor done = high(d) and high(v);

  window join_a = between first and done [1:6];
  window join_b = between second and done [1:M];

  cut cut_between_32 = between join_a and join_b compressed;

  show da = 01 from first until done;
  show high(r) from second until done;
  show stable(tg) from first until done;
  show low(d) before first;
}
