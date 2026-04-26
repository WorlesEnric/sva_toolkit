diagram td_setup_hold_generic_td_000048 {
  clock negedge clk;
  ticks 7;

  lane ack: bit = 0 0 1 0 0 0 0;
  lane stop: bit = 0 0 0 0 1 0 0;
  lane resp_valid: bit = 0 0 0 0 0 0 1;
  lane start: bit = 0 0 0 0 0 0 0;
  lane tag: bus[32] = x x EE x 10 10 10;
  lane valid: bit = 0 0 0 1 0 0 0;
  lane len: bus[32] = 7E EE 00 x x x x;

  anchor launch = rise(ack);
  anchor capture = rise(stop);
  anchor hold_end = high(resp_valid);

  window setup_window = between launch and capture [1:2];
  window hold_window = between capture and hold_end [1:2];

  show low(start) from launch until capture;
  show tag = 10 from capture until hold_end;
  show fall(valid) at capture;
  show change(len) in setup_window;
}
