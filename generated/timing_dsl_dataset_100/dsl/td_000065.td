diagram td_backpressure_interrupt_td_000065 {
  clock posedge clk;
  ticks 14;

  param IRQ_MAX;

  lane irq: bit = 0 1 1 0 0 0 0 0 0 0 0 0 0 0;
  lane pending: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane cause: bus[16] = 1F F0 F0 x x x x x x x x x x x;
  lane clear: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane status: bus[16] = x 02 02 x x x x x x x A5 x x x;
  lane bus_0: bus[8] = x x A5 01 x x x x x x x x x x;
  lane mask: bit = 0 0 1 1 1 1 1 1 1 1 1 1 1 1;

  anchor valid_rise = rise(irq);
  anchor handshake = high(pending) and high(ack);

  window wait_window = between valid_rise and handshake [1:IRQ_MAX];

  show stable(cause) from valid_rise until handshake;
  show low(pending) before valid_rise;
  show high(irq) from valid_rise until handshake;
  show low(clear) from valid_rise until handshake;
  show stable(status) from valid_rise until handshake;
  show bus_0 = A5 at handshake;
  show high(mask) after handshake;
}
