diagram td_chain_interrupt_td_000068 {
  clock posedge clk;
  ticks 17;

  param IRQ_MAX;

  lane status: bus[8] = D0 D0 D0 D0 D0 00 x x x x x x x x x x x;
  lane ack: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane clear: bit = 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane cause: bus[16] = x 01 01 01 x x x x x x D0 x x x x x x;
  lane mask: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane bus_0: bus[8] = x x x x FF 10 x x x x 33 x x x x x x;

  anchor trigger = rise(ack);
  anchor middle = fall(clear);
  anchor response = change(status);

  window first_window = between trigger and middle [1:IRQ_MAX];
  window second_window = between middle and response [2:3];

  show cause = 01 from trigger until middle;
  show low(mask) before response;
  show change(bus_0) at response;
}
