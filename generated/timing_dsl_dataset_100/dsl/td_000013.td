diagram td_burst_interrupt_td_000013 {
  clock negedge clk;
  ticks 9;

  param IRQ_MAX;

  lane cause: bus[8];
  lane mask: bit;
  lane clear: bit;
  lane ack: bit;
  lane irq: bit;
  lane pending: bit;
  lane bit_0: bit;
  lane status: bus[16];
  lane bus_0: bus[8];

  anchor first_beat = high(mask) and high(clear);
  anchor last_beat = high(ack) and high(irq) and eq(cause, 01);
  anchor response_after_last = rise(pending);

  window burst_window = between first_beat and last_beat [1:$];
  window response_after_last_window = between last_beat and response_after_last [1:IRQ_MAX];

  cut cut_after_13 = after last_beat compressed label "gap";

  show stable(bit_0) from first_beat until last_beat;
  show high(ack) from last_beat until response_after_last;
  show low(ack) before first_beat;
  show change(status) at first_beat;
  show change(status) at last_beat;
  show stable(bus_0) in burst_window;
}
