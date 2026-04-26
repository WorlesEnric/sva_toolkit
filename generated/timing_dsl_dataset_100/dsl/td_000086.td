diagram td_setup_hold_noc_td_000086 {
  clock posedge clk;
  ticks 19;

  param CR_MAX;

  lane TX_VALID: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CR_READY: bit = 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RX_VALID: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_HDR: bus[8] = x x 7E 7E 7E x x x x BE x x x x x x x x x;
  lane RX_DATA: bus[32] = x x 1F x x 10 10 10 10 10 10 10 10 10 10 10 10 10 10;

  anchor launch = rise(TX_VALID);
  anchor capture = fall(CR_READY);
  anchor hold_end = high(RX_VALID);

  window setup_window = between launch and capture [2:3];
  window hold_window = between capture and hold_end [1:CR_MAX];

  cut cut_between_86 = between setup_window and hold_window compressed label "compressed lookback context";

  show stable(TX_HDR) from launch until capture;
  show RX_DATA = 10 after hold_end;
}
