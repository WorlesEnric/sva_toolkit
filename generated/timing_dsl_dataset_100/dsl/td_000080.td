diagram td_chain_generic_td_000080 {
  clock posedge clk;
  ticks 8;

  lane req: bit = 0 1 0 0 0 0 0 0;
  lane ready: bit = 0 0 0 1 0 0 0 0;
  lane resp_valid: bit = 0 0 0 1 0 0 0 0;
  lane stop: bit = 0 0 0 0 0 1 0 0;
  lane addr: bus[16] = x A1 A1 A1 x x x 02;
  lane valid: bit = 0 0 0 0 0 0 0 0;

  anchor trigger = rise(req);
  anchor middle = high(ready) and high(resp_valid);
  anchor response = high(stop);

  window first_window = between trigger and middle [2:2];
  window second_window = between middle and response [2:2];

  cut cut_between_80 = between first_window and second_window compressed;

  show stable(addr) from trigger until middle;
  show low(stop) before trigger;
}
