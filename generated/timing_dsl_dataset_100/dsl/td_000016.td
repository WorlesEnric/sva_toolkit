diagram td_fork_compact_td_000016 {
  clock posedge clk;
  ticks 11;

  param M;

  lane v: bit = 1 1 0 0 0 0 0 0 0 0 0;
  lane b: bit = 0 0 0 0 0 1 0 0 0 0 0;
  lane c: bit = 0 0 0 0 0 1 0 0 0 0 0;
  lane d: bit = 1 1 1 1 1 0 0 0 0 0 0;
  lane g: bit = 0 0 1 1 1 1 0 1 0 0 0;
  lane a: bus[16] = FF x 7E 7E 7E 7E x x x x x;
  lane id: bus[32] = x x x x D0 A5 x x x x x;

  anchor trigger = fall(v);
  anchor branch_a = high(b) and high(c);
  anchor branch_b = fall(d);

  window fork_a = between trigger and branch_a [1:M];
  window fork_b = between trigger and branch_b [3:3];

  cut cut_after_16 = after branch_a compressed;

  show high(g) from trigger until branch_a;
  show stable(a) from trigger until branch_a;
  show id = A5 at branch_a;
}
