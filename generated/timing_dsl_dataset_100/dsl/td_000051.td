diagram td_join_compact_td_000051 {
  clock posedge clk;
  ticks 12;

  lane g: bit = 1 1 0 0 0 0 0 0 0 0 0 0;
  lane s: bit = 1 1 0 0 0 0 0 0 0 0 0 0;
  lane v: bit = 0 0 0 0 0 1 0 0 0 0 0 0;
  lane c: bit = 0 0 0 0 0 1 0 0 0 0 0 0;
  lane id: bus[16] = EE 33 x x x x x 0A x x x x;
  lane a: bus[16] = x 33 7E BE EE 00 x x x x x x;
  lane b: bit = 0 0 0 0 0 1 1 1 1 1 1 1;

  anchor first = low(g);
  anchor second = low(s);
  anchor done = high(v) and high(c);

  window join_a = between first and done [1:4];
  window join_b = between second and done [1:4];

  cut cut_before_51 = before first omitted label "before region intentionally hidden for timing context";

  show neq(id, 01) before first;
  show neq(a, 02) in join_a;
  show high(b) after done;
}
