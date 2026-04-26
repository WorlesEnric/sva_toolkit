diagram td_parallel_generic_td_000037 {
  clock posedge clk;
  ticks 17;

  lane addr: bus[16] = BE BE BE BE BE 00 x x x x x x x x x x x;
  lane data: bus[8] = 7E 7E 7E 7E 7E A5 x x x x x x x x x x x;
  lane grant: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane req: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane done: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane start: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane busy: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane len: bus[8] = 7E 01 x x x x x x 0A x x x x x x x x;
  lane id: bus[32] = x x x x x 1F x x x x x x x x x FF x;
  lane stop: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane valid: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor a_start = high(grant);
  anchor a_end = high(ack) and high(req) and eq(addr, 00);
  anchor b_start = rise(done);
  anchor b_end = high(start) and high(busy) and eq(data, A5);

  window a_window = between a_start and a_end [3:3];
  window b_window = between b_start and b_end [3:3];

  show neq(len, F0) before a_start;
  show neq(id, 7E) at a_end;
}
