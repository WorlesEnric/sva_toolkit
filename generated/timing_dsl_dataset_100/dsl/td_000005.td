diagram td_burst_interrupt_td_000005 {
  clock posedge clk;
  ticks 19;

  param IRQ_MAX;

  lane status: bus[8] = 10 10 10 D0 x x x x x x x x x x x x x x x;
  lane mask: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane pending: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane clear: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane irq: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane cause: bus[16] = x 00 7E 0A x x x x x x x x x x x x x A1 x;
  lane bit_0: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane bit_1: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor first_beat = high(mask) and high(pending);
  anchor last_beat = high(ack) and high(clear) and eq(status, D0);
  anchor response_after_last = rise(irq);

  window burst_window = between first_beat and last_beat [1:1];
  window response_after_last_window = between last_beat and response_after_last [1:IRQ_MAX];

  show low(ack) before first_beat;
  show change(cause) at first_beat;
  show change(cause) at last_beat;
  show low(bit_0) at response_after_last;
  show low(bit_1) after response_after_last;
}
