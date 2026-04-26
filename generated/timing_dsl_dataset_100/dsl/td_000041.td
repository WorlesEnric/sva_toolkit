diagram td_burst_noc_td_000041 {
  clock posedge clk;
  ticks 10;

  lane RX_DATA: bus[32] = 00 00 00 00 00 F0 x x x x;
  lane TX_READY: bit = 0 0 1 1 0 0 0 0 0 0;
  lane RX_VALID: bit = 0 0 1 0 0 0 0 0 0 0;
  lane CR_VALID: bit = 0 0 0 1 0 0 0 0 0 0;
  lane CR_READY: bit = 0 0 0 1 0 0 0 0 0 0;
  lane TX_VALID: bit = 0 0 0 0 0 1 0 0 0 0;
  lane RX_READY: bit = 0 0 0 0 0 1 0 0 0 0;
  lane bit_0: bit = 0 0 0 0 0 0 0 1 0 0;
  lane VC: bus[16] = x x x 1F 1F 1F x x x 01;
  lane TX_DATA: bus[16] = x x 02 02 x x 00 x x x;
  lane TX_HDR: bus[8] = x 10 02 A5 BE 55 x D0 x x;

  anchor first_beat = high(TX_READY) and high(RX_VALID);
  anchor middle_beat = high(CR_VALID) and high(CR_READY);
  anchor last_beat = high(TX_VALID) and high(RX_READY) and eq(RX_DATA, F0);
  anchor response_after_last = rise(bit_0);

  window burst_first_to_middle = between first_beat and middle_beat [1:1];
  window burst_window = between middle_beat and last_beat [1:$];
  window response_after_last_window = between last_beat and response_after_last [1:$];

  cut cut_between_41 = between burst_first_to_middle and burst_window lookback label "gap";

  show high(TX_READY) from first_beat until middle_beat;
  show stable(VC) from middle_beat until last_beat;
  show stable(TX_DATA) from first_beat until middle_beat;
  show low(TX_VALID) before first_beat;
  show change(TX_HDR) at first_beat;
  show change(TX_HDR) at middle_beat;
  show change(TX_HDR) at last_beat;
}
