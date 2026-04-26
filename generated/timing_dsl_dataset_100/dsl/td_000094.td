diagram td_chain_fifo_td_000094 {
  clock posedge clk;
  ticks 14;

  lane wreq: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane pop: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane full: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane level: bus[16] = x 55 55 55 0A x x x x x x x x x;
  lane wdata: bus[16] = x EE EE EE x x x x x x x x 01 x;
  lane rdata: bus[8] = EE x x x x x 1F x x x x x x x;
  lane bus_0: bus[8] = x x 02 x x EE EE EE EE EE EE EE EE EE;
  lane empty: bit;
  lane push: bit;

  anchor trigger = high(wreq);
  anchor middle = rise(pop);
  anchor response = rise(full);

  window first_window = between trigger and middle [2:3];
  window second_window = between middle and response [1:4];

  cut cut_after_94 = after response compressed label "next transaction";

  show stable(level) from trigger until middle;
  show stable(wdata) from trigger until middle;
  show neq(rdata, A5) before trigger;
  show stable(bus_0) after response;
}
