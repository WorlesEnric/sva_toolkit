diagram td_parallel_compact_td_000040 {
  clock posedge clk;
  ticks 8;

  param M;

  lane r: bit = 0 1 1 1 0 0 0 0;
  lane s: bit = 1 1 1 0 0 0 0 0;
  lane x: bit = 0 1 0 0 0 0 0 0;
  lane v: bit = 1 1 1 0 0 0 0 0;
  lane id: bus[16] = x x x A5 x x A1 x;
  lane da: bus[16] = x A1 x 1F 1F 1F 1F 1F;

  anchor a_start = rise(r);
  anchor a_end = fall(s);
  anchor b_start = rise(x);
  anchor b_end = low(v);

  window a_window = between a_start and a_end [1:M];
  window b_window = between b_start and b_end [1:2];

  cut cut_after_40 = after a_end compressed;

  show high(r) from a_start until a_end;
  show id = A5 at a_end;
  show stable(da) after b_end;
}
