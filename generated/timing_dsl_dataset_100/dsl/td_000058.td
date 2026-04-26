diagram td_join_memory_td_000058 {
  clock negedge mclk;
  ticks 8;

  param CMD_MAX;

  lane rdata: bus[16] = A5 10 x x x x x x;
  lane wdata: bus[8] = 00 00 00 D0 x x x x;
  lane cmd_ready: bit = 0 1 0 0 0 0 0 0;
  lane rsp_ready: bit = 0 0 0 1 0 0 0 0;
  lane cmd_valid: bit = 0 0 0 1 0 0 0 0;
  lane addr: bus[32] = x x x A5 00 x x x;
  lane rsp_valid: bit = 0 0 0 1 1 1 1 1;
  lane bit_0: bit = 0 0 1 0 0 0 0 0;
  lane bit_1: bit = 0 0 0 0 0 0 0 0;

  anchor first = change(rdata);
  anchor second = rise(cmd_ready);
  anchor done = high(rsp_ready) and high(cmd_valid) and eq(wdata, D0);

  window join_a = between first and done [1:CMD_MAX];
  window join_b = between second and done [2:2];

  show low(rsp_ready) before first;
  show addr = A5 at done;
  show high(rsp_valid) after done;
}
