diagram td_chain_generic_td_000042 {
  clock negedge clk;
  ticks 20;

  lane addr: bus[16] = FF FF FF FF FF FF 0A x x x x x x x x x x x x x;
  lane grant: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ready: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane len: bus[16] = x 0A 0A 0A x x x x x x x x x x x x x x A5 x;
  lane last: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane start: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane stop: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = rise(grant);
  anchor middle = rise(ready);
  anchor response = eq(addr, 0A);

  window first_window = between trigger and middle [1:3];
  window second_window = between middle and response [1:3];

  cut cut_before_42 = before trigger omitted label "next transaction";

  show stable(len) from trigger until middle;
  show low(last) before trigger;
  show low(start) after response;
}
