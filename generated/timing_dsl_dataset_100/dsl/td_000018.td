diagram td_burst_dma_td_000018 {
  clock posedge clk;
  ticks 11;

  lane desc_addr: bus[32] = 00 00 00 00 7E x x x x x x;
  lane dma_start: bit = 0 0 1 1 1 0 0 0 0 0 0;
  lane irq: bit = 0 0 1 0 0 0 0 0 0 0 0;
  lane desc_ready: bit = 0 0 0 0 1 1 1 0 0 0 0;
  lane dma_done: bit = 0 0 0 0 1 0 0 0 0 0 0;
  lane desc_valid: bit = 0 0 0 0 0 0 1 0 0 0 0;
  lane length: bus[32] = 01 x A1 A1 A1 x x x x x x;
  lane status: bus[16] = x A1 0A 33 00 x x BE x x x;
  lane bit_0: bit = 0 0 0 0 0 1 0 0 0 0 0;
  lane bus_0: bus[32] = x x A5 A5 A5 D0 x x x x x;
  lane bit_1: bit = 1 0 0 0 0 0 0 0 0 0 0;

  anchor first_beat = high(dma_start) and high(irq);
  anchor last_beat = high(desc_ready) and high(dma_done) and eq(desc_addr, 7E);
  anchor response_after_last = rise(desc_valid);

  window burst_window = between first_beat and last_beat [2:3];
  window response_after_last_window = between last_beat and response_after_last [2:3];

  cut cut_before_18 = before first_beat omitted label "next transaction";

  show high(dma_start) from first_beat until last_beat;
  show high(desc_ready) from last_beat until response_after_last;
  show stable(length) from first_beat until last_beat;
  show low(desc_ready) before first_beat;
  show change(status) at first_beat;
  show change(status) at last_beat;
  show fall(bit_0) at response_after_last;
  show stable(bus_0) in burst_window;
}
