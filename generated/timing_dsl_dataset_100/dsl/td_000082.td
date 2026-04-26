diagram td_setup_hold_compact_td_000082 {
  clock posedge clk;
  ticks 13;

  param M;

  lane g: bit = 0 0 1 1 1 1 0 0 0 0 0 0 0;
  lane b: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane x: bit = 0 0 0 0 0 0 0 1 0 0 0 0 0;
  lane tg: bus[32] = x EE x x x F0 F0 F0 x x x x x;
  lane a: bus[16] = x x 10 10 10 10 x BE x x x x x;
  lane s: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane r: bit = 0 0 0 0 0 0 0 0 0 0 0 1 0;

  anchor launch = rise(g);
  anchor capture = high(b);
  anchor hold_end = high(x);

  window setup_window = between launch and capture [2:3];
  window hold_window = between capture and hold_end [1:M];

  show high(g) from launch until capture;
  show tg = F0 from capture until hold_end;
  show stable(a) in setup_window;
  show low(x) before launch;
}
