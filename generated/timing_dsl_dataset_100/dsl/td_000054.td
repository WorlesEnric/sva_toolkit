diagram td_parallel_compact_td_000054 {
  clock posedge clk;
  ticks 18;

  lane id: bus[32] = D0 33 D0 33 D0 D0 x x x x x x x x x x x x;
  lane b: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane v: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane x: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane r: bit = 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane g: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane tg: bus[8] = 1F 01 01 01 01 01 x x x x x x x x x x x x;
  lane d: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0;
  lane c: bit;

  anchor a_start = rise(b);
  anchor a_end = high(v) and high(x);
  anchor b_start = low(r);
  anchor b_end = stable(id);

  window a_window = between a_start and a_end [2:5];
  window b_window = between b_start and b_end [1:4];

  show low(g) from a_start until a_end;
  show stable(tg) from a_start until a_end;
  show low(v) before a_start;
  show fall(d) at b_end;
}
