diagram td_fork_axi_like_td_000028 {
  clock negedge ACLK;
  ticks 18;

  param AW_READY_MAX;
  param W_READY_MAX;

  lane WSTRB: bus[32] = A5 A5 A5 0A x x x x x x x x x x x x x x;
  lane ARADDR: bus[8] = 00 00 00 00 A1 x x x x x x x x x x x x x;
  lane AWVALID: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RLAST: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane BREADY: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ARREADY: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane WLAST: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane WDATA: bus[32] = x 7E 7E 7E x x 10 x x x x x x x x x x x;
  lane BVALID: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RVALID: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = rise(AWVALID);
  anchor branch_a = high(RLAST) and high(BREADY) and eq(WSTRB, 0A);
  anchor branch_b = high(ARREADY) and high(WLAST) and eq(ARADDR, A1);

  window fork_a = between trigger and branch_a [1:AW_READY_MAX];
  window fork_b = between trigger and branch_b [1:W_READY_MAX];

  show stable(WDATA) from trigger until branch_a;
  show low(RLAST) before trigger;
}
