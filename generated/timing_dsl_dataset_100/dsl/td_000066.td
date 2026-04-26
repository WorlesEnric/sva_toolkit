diagram td_join_interrupt_td_000066 {
  clock posedge clk;
  ticks 7;

  param IRQ_MAX;

  lane status: bus[8] = 33 33 EE x x x x;
  lane ack: bit = 1 0 0 0 0 0 0;
  lane mask: bit = 0 1 0 0 0 0 0;
  lane cause: bus[8] = 02 EE EE x x x x;
  lane bus_0: bus[8] = x 33 33 x x 7E x;
  lane pending: bit = 0 0 0 0 0 1 0;
  lane irq: bit = 0 1 0 0 0 0 0;
  lane bus_1: bus[8] = 0A FF x A1 x x x;
  lane bus_2: bus[8] = x FF 10 10 10 10 10;

  anchor first = low(ack);
  anchor second = rise(mask);
  anchor done = eq(status, EE);

  window join_a = between first and done [1:2];
  window join_b = between second and done [1:IRQ_MAX];

  show stable(cause) from first until done;
  show stable(bus_0) from first until done;
  show low(pending) before first;
  show fall(irq) at done;
  show change(bus_1) in join_a;
  show bus_2 = 10 after done;
}
