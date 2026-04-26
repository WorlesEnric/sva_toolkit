diagram td_fork_generic_td_000098 {
  clock negedge clk;
  ticks 7;

  lane ready: bit = 0 0 1 0 0 0 0;
  lane start: bit = 0 0 0 0 1 0 0;
  lane req: bit = 0 0 0 0 1 0 0;
  lane stop: bit = 0 0 0 0 1 0 0;
  lane grant: bit = 0 0 0 0 0 0 0;
  lane tag: bus[16] = x x F0 F0 F0 x 7E;
  lane busy: bit;

  anchor trigger = rise(ready);
  anchor branch_a = high(start) and high(req);
  anchor branch_b = high(stop);

  window fork_a = between trigger and branch_a [1:2];
  window fork_b = between trigger and branch_b [1:2];

  show stable(grant) from trigger until branch_a;
  show stable(tag) in fork_b;
}
