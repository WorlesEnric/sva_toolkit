diagram td_burst_generic_td_000089 {
  clock posedge clk;
  ticks 18;

  lane len: bus[32];
  lane stop: bit;
  lane grant: bit;
  lane last: bit;
  lane req: bit;
  lane resp_valid: bit;
  lane data: bus[8];
  lane start: bit;

  anchor first_beat = high(stop) and high(grant);
  anchor last_beat = high(last) and high(req) and eq(len, 10);
  anchor response_after_last = rise(resp_valid);

  window burst_window = between first_beat and last_beat [1:1];
  window response_after_last_window = between last_beat and response_after_last [2:2];

  cut cut_between_89 = between burst_window and response_after_last_window lookback label "gap";

  show low(last) before first_beat;
  show change(data) at first_beat;
  show change(data) at last_beat;
  show high(start) after response_after_last;
}
