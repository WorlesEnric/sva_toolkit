diagram td_backpressure_interrupt_td_000030 {
  clock posedge clk;
  ticks 7;

  lane clear: bit;
  lane mask: bit;
  lane irq: bit;
  lane status: bus[16];
  lane cause: bus[16];
  lane ack: bit;
  lane bus_0: bus[8];
  lane bus_1: bus[8];
  lane bus_2: bus[16];

  anchor valid_rise = rise(clear);
  anchor handshake = high(mask) and high(irq);

  window wait_window = between valid_rise and handshake [3:3];

  show high(clear) from valid_rise until handshake;
  show stable(status) from valid_rise until handshake;
  show neq(cause, 00) before valid_rise;
  show high(clear) from valid_rise until handshake;
  show low(ack) from valid_rise until handshake;
  show stable(bus_0) from valid_rise until handshake;
  show neq(bus_1, 7E) at handshake;
  show change(bus_2) in wait_window;
}
