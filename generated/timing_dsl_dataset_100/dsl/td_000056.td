diagram td_burst_noc_td_000056 {
  clock negedge clk;
  ticks 12;

  lane RX_DATA: bus[32] = 01 01 01 01 10 x x x x x x x;
  lane TX_VALID: bit = 0 1 0 0 0 0 0 0 0 0 0 0;
  lane RX_VALID: bit = 0 1 0 0 0 0 0 0 0 0 0 0;
  lane TX_READY: bit = 0 0 0 0 1 0 0 0 0 0 0 0;
  lane CR_READY: bit = 0 0 0 0 1 0 0 0 0 0 0 0;
  lane RX_READY: bit = 0 0 0 0 0 0 0 0 1 0 0 0;
  lane VC: bus[16] = x FF FF FF FF x x A1 x x x x;
  lane TX_HDR: bus[8] = 10 0A x x x x x x x x x x;
  lane TX_DATA: bus[32] = D0 55 x 1F 02 x 01 x x x x x;

  anchor first_beat = high(TX_VALID) and high(RX_VALID);
  anchor last_beat = high(TX_READY) and high(CR_READY) and eq(RX_DATA, 10);
  anchor response_after_last = rise(RX_READY);

  window burst_window = between first_beat and last_beat [1:3];
  window response_after_last_window = between last_beat and response_after_last [2:4];

  cut cut_between_56 = between burst_window and response_after_last_window compressed;

  show stable(VC) from first_beat until last_beat;
  show neq(TX_HDR, 7E) before first_beat;
  show change(TX_DATA) at first_beat;
  show change(TX_DATA) at last_beat;
}
