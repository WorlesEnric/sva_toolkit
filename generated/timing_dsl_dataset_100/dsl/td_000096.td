diagram td_setup_hold_memory_td_000096 {
  clock posedge mclk;
  ticks 9;

  param CMD_MAX;

  lane wdata: bus[32] = 0A 0A A5 x x x x x x;
  lane rsp_ready: bit = 0 0 0 0 0 1 0 0 0;
  lane cmd_valid: bit = 0 0 0 0 0 0 0 1 0;
  lane rdata: bus[16] = x x A5 A5 A5 A5 x 10 x;
  lane addr: bus[16] = 10 x 00 00 00 00 x x x;
  lane bus_0: bus[32] = x x x x x BE x 0A x;
  lane cmd_ready: bit = 0 0 0 0 0 0 0 0 0;
  lane rsp_valid: bit;

  anchor launch = change(wdata);
  anchor capture = rise(rsp_ready);
  anchor hold_end = high(cmd_valid);

  window setup_window = between launch and capture [3:3];
  window hold_window = between capture and hold_end [1:CMD_MAX];

  show stable(rdata) from launch until capture;
  show stable(addr) from launch until capture;
  show low(cmd_valid) before launch;
  show neq(bus_0, EE) at capture;
  show low(cmd_ready) after hold_end;
}
