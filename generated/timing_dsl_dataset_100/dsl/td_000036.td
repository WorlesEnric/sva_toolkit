diagram td_burst_interrupt_td_000036 {
  clock negedge clk;
  ticks 17;

  lane status: bus[8] = 33 33 33 33 F0 x x x x x x x x x x x x;
  lane clear: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane mask: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane pending: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane irq: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane bit_0: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane cause: bus[8] = x x 10 10 x x x x 0A x x x x x x x x;
  lane bus_0: bus[8] = x x A1 A1 x x x x x x x x FF x x x x;
  lane bus_1: bus[16] = x x 00 00 x A1 x x x x x x x x x x x;
  lane bus_2: bus[16] = x EE 10 FF 02 x x x x x x x x x 0A x x;

  anchor first_beat = high(clear) and high(ack);
  anchor middle_beat = high(mask) and high(pending);
  anchor last_beat = high(irq) and high(bit_0) and eq(status, F0);

  window burst_first_to_middle = between first_beat and middle_beat [1:1];
  window burst_window = between middle_beat and last_beat [1:1];

  show stable(cause) from first_beat until middle_beat;
  show stable(bus_0) from first_beat until middle_beat;
  show stable(bus_1) from first_beat until middle_beat;
  show change(bus_2) at first_beat;
  show change(bus_2) at middle_beat;
  show change(bus_2) at last_beat;
}
