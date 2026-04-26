diagram td_fork_memory_td_000039 {
  clock posedge mclk;
  ticks 19;

  param CMD_MAX;
  param RSP_MAX;

  lane rsp_ready: bit = 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane rsp_valid: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane cmd_valid: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane cmd_ready: bit = 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane wdata: bus[16] = x x 0A 0A 0A x 1F x x x x x x x x x x x x;
  lane addr: bus[32] = x x 55 55 x x x x 33 x x x x x x x x x x;
  lane rdata: bus[8] = x x 33 33 33 x x x x x x x x A5 x x x x x;
  lane bit_0: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = fall(rsp_ready);
  anchor branch_a = high(rsp_valid);
  anchor branch_b = rise(cmd_valid);

  window fork_a = between trigger and branch_a [1:CMD_MAX];
  window fork_b = between trigger and branch_b [1:RSP_MAX];

  cut cut_before_39 = before trigger omitted label "before region intentionally hidden for timing context";

  show high(cmd_ready) from trigger until branch_a;
  show wdata = 0A from trigger until branch_b;
  show stable(addr) from trigger until branch_a;
  show rdata = 33 in fork_b;
}
