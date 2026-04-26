diagram td_burst_dma_td_000026 {
  clock negedge clk;
  ticks 9;

  param DESC_MAX;

  lane status: bus[32] = EE EE EE EE BE x x x x;
  lane dma_start: bit = 0 1 0 0 0 0 0 0 0;
  lane desc_valid: bit = 0 1 0 0 0 0 0 0 0;
  lane dma_done: bit = 0 0 0 0 1 0 0 0 0;
  lane irq: bit = 0 0 0 0 1 0 0 0 0;
  lane desc_ready: bit = 0 0 0 0 0 0 1 0 0;
  lane bit_0: bit = 0 1 1 1 1 0 0 0 0;
  lane bit_1: bit = 0 0 0 0 0 0 0 0 0;
  lane length: bus[32] = 1F EE x A1 55 33 x x x;

  anchor first_beat = high(dma_start) and high(desc_valid);
  anchor last_beat = high(dma_done) and high(irq) and eq(status, BE);
  anchor response_after_last = rise(desc_ready);

  window burst_window = between first_beat and last_beat [1:DESC_MAX];
  window response_after_last_window = between last_beat and response_after_last [1:3];

  show stable(bit_0) from first_beat until last_beat;
  show low(bit_1) before response_after_last;
  show change(length) at first_beat;
  show change(length) at last_beat;
}
