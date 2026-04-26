diagram td_backpressure_interrupt_td_000091 {
  clock posedge clk;
  ticks 15;

  lane clear: bit = 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0;
  lane irq: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane cause: bus[16] = x x D0 D0 D0 7E x x x x x x x x x;
  lane status: bus[8] = x x 01 01 01 x x x x BE x x x x x;
  lane pending: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane bus_0: bus[16] = x x EE EE EE 0A x x x x x x x x x;
  lane bus_1: bus[8] = x x 02 02 02 BE x x x x x x x x x;

  anchor valid_rise = rise(clear);
  anchor handshake = high(irq) and high(ack);

  window wait_window = between valid_rise and handshake [2:5];

  cut cut_after_91 = after handshake compressed label "after region intentionally hidden for timing context";

  show high(clear) from valid_rise until handshake;
  show stable(cause) in wait_window;
  show stable(status) in wait_window;
  show high(clear) from valid_rise until handshake;
  show low(pending) from valid_rise until handshake;
  show stable(bus_0) from valid_rise until handshake;
  show bus_1 = 02 in wait_window;
}
