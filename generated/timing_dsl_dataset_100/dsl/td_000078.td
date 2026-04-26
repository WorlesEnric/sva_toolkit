diagram td_burst_generic_td_000078 {
  clock posedge clk;
  ticks 16;

  lane len: bus[32];
  lane resp_valid: bit;
  lane grant: bit;
  lane ack: bit;
  lane last: bit;
  lane stop: bit;
  lane busy: bit;
  lane valid: bit;
  lane tag: bus[32];
  lane data: bus[32];
  lane addr: bus[16];

  anchor first_beat = high(resp_valid) and high(grant);
  anchor middle_beat = high(ack) and high(last);
  anchor last_beat = high(stop) and high(busy) and eq(len, A1);
  anchor response_after_last = rise(valid);

  window burst_first_to_middle = between first_beat and middle_beat [2:3];
  window burst_window = between middle_beat and last_beat [2:3];
  window response_after_last_window = between last_beat and response_after_last [2:4];

  show stable(tag) in burst_first_to_middle;
  show low(stop) before first_beat;
  show change(data) at first_beat;
  show change(data) at middle_beat;
  show change(data) at last_beat;
  show addr = 33 after response_after_last;
}
