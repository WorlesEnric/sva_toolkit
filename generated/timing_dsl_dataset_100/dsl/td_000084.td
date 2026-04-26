diagram td_parallel_generic_td_000084 {
  clock posedge clk;
  ticks 17;

  lane resp_valid: bit = 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane stop: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ready: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane done: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane req: bit = 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1;

  anchor a_start = high(resp_valid);
  anchor a_end = high(stop);
  anchor b_start = rise(ready);
  anchor b_end = rise(done);

  window a_window = between a_start and a_end [2:4];
  window b_window = between b_start and b_end [2:3];

  show high(resp_valid) from a_start until a_end;
  show low(stop) before a_start;
  show high(req) after b_end;
}
