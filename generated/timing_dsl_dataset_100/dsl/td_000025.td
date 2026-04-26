diagram td_fork_fifo_td_000025 {
  clock posedge clk;
  ticks 15;

  param FIFO_MAX;

  lane push: bit = 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane wreq: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane rreq: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane pop: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane level: bus[16] = x x 01 01 01 01 01 x EE x x x x x x;
  lane empty: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane full: bit = 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;

  anchor trigger = low(push);
  anchor branch_a = high(wreq) and high(rreq);
  anchor branch_b = high(pop);

  window fork_a = between trigger and branch_a [2:5];
  window fork_b = between trigger and branch_b [1:FIFO_MAX];

  cut cut_after_25 = after branch_a compressed label "gap";

  show stable(level) from trigger until branch_a;
}
