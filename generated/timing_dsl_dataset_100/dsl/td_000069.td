diagram td_parallel_generic_td_000069 {
  clock negedge clk;
  ticks 10;

  lane addr: bus[16] = F0 F0 10 x x x x x x x;
  lane tag: bus[8] = 02 02 02 02 02 BE x x x x;
  lane id: bus[16] = 10 10 1F x x x x x x x;
  lane ack: bit = 0 0 0 0 0 1 0 0 0 0;
  lane busy: bit = 0 0 0 0 0 1 0 0 0 0;
  lane valid: bit = 0 0 0 0 1 0 0 0 0 0;
  lane last: bit = 0 0 1 1 0 0 0 0 0 0;
  lane len: bus[8] = x x 10 10 10 x x x 01 x;

  anchor a_start = change(addr);
  anchor a_end = high(ack) and high(busy) and eq(tag, BE);
  anchor b_start = change(id);
  anchor b_end = rise(valid);

  window a_window = between a_start and a_end [2:3];
  window b_window = between b_start and b_end [2:3];

  show low(ack) before a_start;
  show fall(last) at b_end;
  show len = 10 in b_window;
}
