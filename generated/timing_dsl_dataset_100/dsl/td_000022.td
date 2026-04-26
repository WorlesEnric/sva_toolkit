diagram td_fork_memory_td_000022 {
  clock posedge mclk;
  ticks 9;

  lane rdata: bus[32] = 00 00 00 00 33 x x x x;
  lane cmd_valid: bit = 0 0 1 0 0 0 0 0 0;
  lane cmd_ready: bit = 0 0 0 1 0 0 0 0 0;
  lane rsp_ready: bit = 0 0 0 0 1 0 0 0 0;
  lane rsp_valid: bit = 0 0 0 0 1 0 0 0 0;
  lane bit_0: bit = 0 0 0 0 0 0 0 0 0;
  lane addr: bus[16] = FF FF x 02 x x x x x;
  lane bit_1: bit = 0 0 0 1 1 0 0 0 0;
  lane bit_2: bit = 0 0 0 0 1 1 1 1 1;

  anchor trigger = rise(cmd_valid);
  anchor branch_a = rise(cmd_ready);
  anchor branch_b = high(rsp_ready) and high(rsp_valid) and eq(rdata, 33);

  window fork_a = between trigger and branch_a [1:3];
  window fork_b = between trigger and branch_b [2:3];

  show low(bit_0) from trigger until branch_a;
  show neq(addr, 10) before trigger;
  show high(bit_1) at branch_a;
  show high(bit_2) after branch_b;
}
