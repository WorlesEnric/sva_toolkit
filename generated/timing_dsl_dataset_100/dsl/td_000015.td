diagram td_parallel_generic_td_000015 {
  clock posedge clk;
  ticks 8;

  param MAX_LAT;

  lane resp_valid: bit = 0 1 0 0 0 0 0 0;
  lane stop: bit = 0 0 0 1 0 0 0 0;
  lane done: bit = 1 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 0 1 0 0 0 0;

  anchor a_start = high(resp_valid);
  anchor a_end = rise(stop);
  anchor b_start = fall(done);
  anchor b_end = rise(ack);

  window a_window = between a_start and a_end [1:MAX_LAT];
  window b_window = between b_start and b_end [2:2];

  show low(stop) before a_start;
}
