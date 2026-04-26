diagram td_setup_hold_fifo_td_000024 {
  clock negedge clk;
  ticks 14;

  lane wreq: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane empty: bit = 0 0 0 0 1 1 0 0 0 0 0 0 0 0;
  lane pop: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane wdata: bus[16] = x 0A 0A 0A 0A x x x x x x D0 x x;
  lane full: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor launch = rise(wreq);
  anchor capture = rise(empty);
  anchor hold_end = high(pop);

  window setup_window = between launch and capture [3:3];
  window hold_window = between capture and hold_end [1:1];

  show stable(wdata) from launch until capture;
  show high(empty) from capture until hold_end;
  show low(full) before hold_end;
}
