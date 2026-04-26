diagram td_fork_noc_td_000087 {
  clock posedge clk;
  ticks 17;

  lane TX_DATA: bus[8] = 33 33 1F x x x x x x x x x x x x x x;
  lane CR_VALID: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CR_READY: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_HDR: bus[8] = x 7E 0A 0A x x x x x x x x x x x x x;
  lane RX_DATA: bus[8] = x x 0A 0A x x x x x x x 00 x x x x x;
  lane RX_VALID: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = change(TX_DATA);
  anchor branch_a = high(CR_VALID);
  anchor branch_b = rise(CR_READY);

  window fork_a = between trigger and branch_a [1:1];
  window fork_b = between trigger and branch_b [2:2];

  show stable(TX_HDR) from trigger until branch_a;
  show stable(RX_DATA) from trigger until branch_a;
  show low(RX_VALID) after branch_b;
}
