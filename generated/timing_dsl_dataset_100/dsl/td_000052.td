diagram td_chain_generic_td_000052 {
  clock negedge clk;
  ticks 20;

  param MAX_LAT;

  lane grant: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ready: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane tag: bus[32] = x 7E 7E x x x x x FF x x x x x x x x x x x;
  lane len: bus[32] = x D0 D0 x x x x x x x x x x x x x x 0A x x;
  lane last: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane valid: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = rise(grant);
  anchor middle = rise(ready);
  anchor response = rise(ack);

  window first_window = between trigger and middle [1:MAX_LAT];
  window second_window = between middle and response [1:1];

  show stable(tag) from trigger until middle;
  show stable(len) from trigger until middle;
  show low(ack) before trigger;
}
