diagram td_chain_compact_td_000049 {
  clock posedge clk;
  ticks 9;

  lane c: bit = 1 1 0 0 0 0 0 0 0;
  lane b: bit = 0 0 0 1 0 0 0 0 0;
  lane s: bit = 0 0 0 0 0 0 1 0 0;
  lane da: bus[32] = x x 33 33 x x x 10 x;
  lane g: bit = 0 0 0 0 0 0 0 0 0;
  lane x: bit = 0 0 0 0 0 0 0 0 0;

  anchor trigger = low(c);
  anchor middle = rise(b);
  anchor response = rise(s);

  window first_window = between trigger and middle [1:1];
  window second_window = between middle and response [3:3];

  cut cut_after_49 = after response compressed label "gap";

  show stable(da) from trigger until middle;
  show low(g) before response;
  show low(x) after response;
}
