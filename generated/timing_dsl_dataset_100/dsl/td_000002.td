diagram td_fork_fifo_td_000002 {
  clock posedge clk;
  ticks 13;

  lane wdata: bus[8] = EE EE EE EE EE 10 x x x x x x x;
  lane full: bit = 0 0 1 1 1 1 0 0 0 0 0 0 0;
  lane wreq: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane pop: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane rreq: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane push: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane level: bus[8] = x x 33 33 33 33 x x x x A1 x x;
  lane empty: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = rise(full);
  anchor branch_a = high(wreq) and high(pop) and eq(wdata, 10);
  anchor branch_b = rise(rreq);

  window fork_a = between trigger and branch_a [2:3];
  window fork_b = between trigger and branch_b [1:2];

  cut cut_between_2 = between fork_a and fork_b compressed label "compressed lookback context";

  show high(full) from trigger until branch_a;
  show low(wreq) before trigger;
  show low(push) at branch_a;
  show stable(level) in fork_a;
}
