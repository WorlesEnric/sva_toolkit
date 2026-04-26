diagram td_backpressure_interrupt_td_000008 {
  clock negedge clk;
  ticks 12;

  lane ack: bit = 0 0 1 1 1 1 1 1 0 0 0 0;
  lane irq: bit = 0 0 0 0 0 0 0 1 0 0 0 0;
  lane clear: bit = 0 0 0 0 0 0 0 1 0 0 0 0;
  lane cause: bus[8] = x x EE EE EE EE EE EE 55 x x x;
  lane status: bus[8] = x x 1F 1F 1F 1F 1F 1F x x EE x;
  lane bus_0: bus[8] = x x 33 33 33 33 33 33 1F x x x;
  lane pending: bit = 0 0 0 0 0 0 0 0 0 0 0 0;
  lane bus_1: bus[16] = x x EE EE EE EE EE EE x A5 x x;
  lane bus_2: bus[8] = x 0A 00 00 00 00 00 00 x x x x;

  anchor valid_rise = rise(ack);
  anchor handshake = high(irq) and high(clear);

  window wait_window = between valid_rise and handshake [2:5];

  show stable(cause) from valid_rise until handshake;
  show stable(status) from valid_rise until handshake;
  show stable(bus_0) in wait_window;
  show low(irq) before valid_rise;
  show high(ack) from valid_rise until handshake;
  show low(pending) from valid_rise until handshake;
  show stable(bus_1) from valid_rise until handshake;
  show stable(bus_2) in wait_window;
}
