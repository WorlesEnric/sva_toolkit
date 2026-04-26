diagram td_fork_compact_td_000077 {
  clock negedge clk;
  ticks 9;

  param M;

  lane a: bus[16] = 01 01 01 02 x x x x x;
  lane s: bit = 0 0 1 0 0 0 0 0 0;
  lane g: bit = 0 0 0 1 0 0 0 0 0;
  lane x: bit = 0 0 0 1 0 0 0 0 0;
  lane d: bit = 0 0 0 0 1 0 0 0 0;
  lane tg: bus[8] = 33 x 1F 1F x x x x x;
  lane da: bus[8] = 00 EE x x D0 x x x x;
  lane id: bus[8] = x x 0A x 02 02 02 02 02;

  anchor trigger = rise(s);
  anchor branch_a = high(g) and high(x) and eq(a, 02);
  anchor branch_b = rise(d);

  window fork_a = between trigger and branch_a [1:$];
  window fork_b = between trigger and branch_b [1:M];

  cut cut_between_77 = between fork_a and fork_b lookback label "gap";

  show stable(tg) from trigger until branch_a;
  show neq(da, 0A) before trigger;
  show id = 02 after branch_b;
}
