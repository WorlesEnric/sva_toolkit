diagram td_parallel_compact_td_000009 {
  clock posedge clk;
  ticks 9;

  lane a: bus[32] = A1 A1 x x x x x x x;
  lane id: bus[8] = BE 0A BE BE x x x x x;
  lane tg: bus[16] = A1 00 x x x x x x x;
  lane da: bus[16] = 10 F0 10 10 x x x x x;
  lane s: bit = 0 1 1 1 0 0 0 0 0;
  lane d: bit = 0 1 1 1 0 0 0 0 0;
  lane bus_0: bus[32] = x A5 A5 A5 x x 1F x x;

  anchor a_start = stable(a);
  anchor a_end = stable(id);
  anchor b_start = neq(tg, A1);
  anchor b_end = stable(da);

  window a_window = between a_start and a_end [2:3];
  window b_window = between b_start and b_end [2:3];

  show high(s) from a_start until a_end;
  show high(d) from b_start until b_end;
  show stable(bus_0) in b_window;
}
