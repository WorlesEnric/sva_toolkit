diagram td_burst_dma_td_000097 {
  clock posedge clk;
  ticks 8;

  param DESC_MAX;

  lane status: bus[16] = 02 02 A1 x x x x x;
  lane desc_valid: bit = 0 1 0 0 0 0 0 0;
  lane dma_done: bit = 0 1 0 0 0 0 0 0;
  lane desc_ready: bit = 0 0 1 0 0 0 0 0;
  lane dma_start: bit = 0 0 1 0 0 0 0 0;
  lane irq: bit = 0 0 0 1 0 0 0 0;
  lane desc_addr: bus[16] = 01 A5 x x x x x x;
  lane length: bus[16] = 55 BE 55 10 x x x x;
  lane bus_0: bus[16] = x x 01 01 x x x FF;
  lane bit_0: bit = 0 1 0 0 0 0 0 0;
  lane bit_1: bit;

  anchor first_beat = high(desc_valid) and high(dma_done);
  anchor last_beat = high(desc_ready) and high(dma_start) and eq(status, A1);
  anchor response_after_last = rise(irq);

  window burst_window = between first_beat and last_beat [1:DESC_MAX];
  window response_after_last_window = between last_beat and response_after_last [1:2];

  cut cut_after_97 = after last_beat compressed label "gap";

  show neq(desc_addr, 1F) before first_beat;
  show change(length) at first_beat;
  show change(length) at last_beat;
  show bus_0 = 01 in response_after_last_window;
  show low(bit_0) after response_after_last;
}
