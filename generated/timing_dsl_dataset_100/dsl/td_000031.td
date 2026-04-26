diagram td_fork_fifo_td_000031 {
  clock negedge clk;
  ticks 15;

  lane rdata: bus[16] = D0 D0 02 x x x x x x x x x x x x;
  lane wdata: bus[8] = FF FF FF FF FF BE x x x x x x x x x;
  lane pop: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane rreq: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane empty: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane wreq: bit = 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane level: bus[16] = x x FF FF x x x x 10 x x x x x x;

  anchor trigger = change(rdata);
  anchor branch_a = rise(pop);
  anchor branch_b = high(rreq) and high(empty) and eq(wdata, BE);

  window fork_a = between trigger and branch_a [1:4];
  window fork_b = between trigger and branch_b [2:3];

  show high(wreq) from trigger until branch_a;
  show stable(level) in fork_a;
}
