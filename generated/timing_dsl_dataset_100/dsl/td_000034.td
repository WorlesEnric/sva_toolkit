diagram td_chain_generic_td_000034 {
  clock negedge clk;
  ticks 15;

  lane addr: bus[16] = 7E 7E 7E 7E 55 x x x x x x x x x x;
  lane busy: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane start: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane done: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane grant: bit = 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane data: bus[8] = x 10 x x x x x BE x x x x x x x;
  lane len: bus[32] = x x x x 55 55 55 55 BE x x x x x x;

  anchor trigger = rise(busy);
  anchor middle = high(start) and high(done) and eq(addr, 55);
  anchor response = high(grant);

  window first_window = between trigger and middle [1:3];
  window second_window = between middle and response [2:4];

  show neq(data, EE) at response;
  show len = 55 in second_window;
}
