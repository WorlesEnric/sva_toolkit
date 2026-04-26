diagram td_join_compact_td_000043 {
  clock posedge clk;
  ticks 18;

  param M;
  param N;

  lane s: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane g: bit = 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane r: bit = 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane da: bus[16] = x x F0 F0 F0 F0 BE x x x x x x x x x x x;
  lane id: bus[16] = x x F0 F0 F0 F0 x x x x x x x x x x 55 x;
  lane a: bus[32] = x x 33 33 33 33 x x x x x x x x x x x FF;
  lane b: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor first = rise(s);
  anchor second = rise(g);
  anchor done = low(r);

  window join_a = between first and done [1:M];
  window join_b = between second and done [1:N];

  show da = F0 from first until done;
  show high(g) from second until done;
  show stable(id) in join_a;
  show stable(a) in join_a;
  show low(b) before done;
}
