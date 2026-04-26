diagram td_chain_generic_td_000064 {
  clock posedge clk;
  ticks 18;

  lane done: bit = 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane grant: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane resp_valid: bit = 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0;
  lane addr: bus[32] = x F0 F0 F0 F0 x 33 x x x x x x x x x x x;
  lane id: bus[32] = 33 x x x x x x x x EE x x x x x x x x;
  lane ack: bit = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane last: bit = 0 0 0 1 0 0 0 0 1 1 1 1 1 1 1 1 1 1;

  anchor trigger = fall(done);
  anchor middle = rise(grant);
  anchor response = fall(resp_valid);

  window first_window = between trigger and middle [2:4];
  window second_window = between middle and response [2:4];

  show stable(addr) from trigger until middle;
  show neq(id, D0) before trigger;
  show high(ack) at response;
  show high(last) after response;
}
