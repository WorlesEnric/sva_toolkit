diagram td_setup_hold_compact_td_000007 {
  clock negedge clk;
  ticks 11;

  lane id: bus[8] = 01 01 EE x x x x x x x x;
  lane v: bit = 0 0 0 0 1 0 0 0 0 0 0;
  lane a: bus[16] = x x BE BE BE x x x FF x x;
  lane d: bit = 0 0 0 0 1 0 0 0 0 0 0;
  lane r: bit = 0 0 0 0 0 0 0 0 0 0 0;
  lane s: bit = 0 0 0 0 0 0 0 1 0 0 0;
  lane g: bit = 0 0 0 0 0 0 0 0 0 0 0;

  anchor capture = neq(id, 01);
  anchor hold_end = high(v);

  window hold_window = between capture and hold_end [1:3];

  show stable(a) from capture until hold_end;
  show high(d) at hold_end;
  show low(r) after hold_end;
}
