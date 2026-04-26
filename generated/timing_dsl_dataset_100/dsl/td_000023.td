diagram td_backpressure_interrupt_td_000023 {
  clock negedge clk;
  ticks 7;

  lane mask: bit;
  lane irq: bit;
  lane clear: bit;
  lane ack: bit;
  lane cause: bus[16];
  lane status: bus[16];

  anchor valid_rise = rise(mask);
  anchor handshake = high(irq) and high(clear);

  window wait_window = between valid_rise and handshake [3:3];

  show low(irq) before valid_rise;
  show high(mask) from valid_rise until handshake;
  show low(ack) from valid_rise until handshake;
  show stable(cause) from valid_rise until handshake;
  show status = EE at handshake;
}
