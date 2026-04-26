diagram td_fork_noc_td_000057 {
  clock negedge clk;
  ticks 12;

  param CR_MAX;
  param TX_MAX;

  lane TX_READY: bit = 0 0 1 0 0 0 0 0 0 0 0 0;
  lane TX_VALID: bit = 0 0 0 1 0 0 0 0 0 0 0 0;
  lane CR_READY: bit = 0 0 0 0 1 0 0 0 0 0 0 0;
  lane CR_VALID: bit = 0 0 0 0 1 0 0 0 0 0 0 0;
  lane RX_DATA: bus[32] = x x FF FF x x x 00 x x x x;

  anchor trigger = rise(TX_READY);
  anchor branch_a = rise(TX_VALID);
  anchor branch_b = high(CR_READY) and high(CR_VALID);

  window fork_a = between trigger and branch_a [1:CR_MAX];
  window fork_b = between trigger and branch_b [1:TX_MAX];

  cut cut_before_57 = before trigger omitted label "idle";

  show stable(RX_DATA) in fork_a;
}
