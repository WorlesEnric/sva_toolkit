diagram td_setup_hold_interrupt_td_000067 {
  clock negedge clk;
  ticks 10;

  lane ack: bit = 0 0 1 0 0 0 0 0 0 0;
  lane mask: bit = 0 0 0 0 0 1 0 0 0 0;
  lane pending: bit = 0 0 0 0 0 1 0 0 0 0;
  lane irq: bit = 0 0 0 0 0 0 0 1 0 0;
  lane status: bus[16] = x x A5 A5 A5 A5 x x x 33;
  lane cause: bus[16] = x x D0 D0 D0 D0 x x 33 x;
  lane clear: bit = 0 0 0 0 0 0 0 0 0 0;
  lane bus_0: bus[8] = x x x x BE x x 0A 0A 0A;
  lane bit_0: bit = 0 0 0 0 0 0 0 0 0 0;

  anchor launch = rise(ack);
  anchor capture = high(mask) and high(pending);
  anchor hold_end = high(irq);

  window setup_window = between launch and capture [2:3];
  window hold_window = between capture and hold_end [2:3];

  cut cut_after_67 = after hold_end compressed label "after region intentionally hidden for timing context";

  show status = A5 from launch until capture;
  show stable(cause) in setup_window;
  show low(clear) before hold_end;
  show stable(bus_0) after hold_end;
}
