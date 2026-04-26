diagram td_parallel_dma_td_000004 {
  clock posedge clk;
  ticks 6;

  lane dma_done: bit = 0 0 1 0 0 0;
  lane desc_ready: bit = 0 0 0 0 1 0;
  lane irq: bit = 1 1 0 0 0 0;
  lane desc_valid: bit = 0 0 0 1 0 0;
  lane desc_addr: bus[16] = 02 x x EE EE EE;
  lane dma_start: bit = 0 0 0 0 0 0;

  anchor a_start = high(dma_done);
  anchor a_end = rise(desc_ready);
  anchor b_start = low(irq);
  anchor b_end = high(desc_valid);

  window a_window = between a_start and a_end [2:2];
  window b_window = between b_start and b_end [1:1];

  show desc_addr = EE after b_end;
}
