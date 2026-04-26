diagram td_burst_generic_td_000074 {
  clock negedge clk;
  ticks 12;

  param MAX_LAT;

  lane id: bus[32] = A1 A1 A1 01 x x x x x x x x;
  lane ready: bit = 0 1 0 0 0 0 0 0 0 0 0 0;
  lane grant: bit = 0 1 0 0 0 0 0 0 0 0 0 0;
  lane stop: bit = 0 0 0 1 0 0 0 0 0 0 0 0;
  lane start: bit = 0 0 0 1 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 0 0 0 0 1 0 0 0 0 0;
  lane tag: bus[32] = x 55 55 55 x x x x x x x 1F;
  lane data: bus[32] = D0 FF A1 1F x x x x 01 x x x;
  lane resp_valid: bit = 0 0 0 0 1 0 1 1 1 1 1 1;
  lane valid: bit = 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor first_beat = high(ready) and high(grant);
  anchor last_beat = high(stop) and high(start) and eq(id, 01);
  anchor response_after_last = rise(ack);

  window burst_window = between first_beat and last_beat [1:MAX_LAT];
  window response_after_last_window = between last_beat and response_after_last [3:3];

  cut cut_between_74 = between burst_window and response_after_last_window compressed label "compressed lookback context";

  show stable(tag) from first_beat until last_beat;
  show change(data) at first_beat;
  show change(data) at last_beat;
  show high(resp_valid) after response_after_last;
}
