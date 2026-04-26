diagram td_parallel_generic_td_000099 {
  clock posedge clk;
  ticks 13;

  lane tag: bus[8] = FF FF FF FF FF 00 x x x x x x x;
  lane stop: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane last: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane valid: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane req: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane start: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane done: bit = 0 0 0 0 0 0 0 0 0 1 0 0 0;
  lane len: bus[16] = x x 1F A5 BE FF x x x x x x 10;
  lane data: bus[16] = F0 x x x x 0A 0A 0A 0A 0A 0A 0A 0A;
  lane resp_valid: bit;
  lane ack: bit;

  anchor a_start = rise(stop);
  anchor a_end = high(last);
  anchor b_start = high(valid);
  anchor b_end = high(req) and high(start) and eq(tag, 00);

  window a_window = between a_start and a_end [2:2];
  window b_window = between b_start and b_end [3:3];

  cut cut_before_99 = before a_start omitted label "before region intentionally hidden for timing context";

  show low(done) from a_start until a_end;
  show low(last) before a_start;
  show neq(len, A1) in b_window;
  show data = 0A after b_end;
}
