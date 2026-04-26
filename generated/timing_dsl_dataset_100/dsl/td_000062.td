diagram td_join_dma_td_000062 {
  clock negedge clk;
  ticks 17;

  lane dma_start: bit;
  lane irq: bit;
  lane desc_ready: bit;
  lane status: bus[16];
  lane desc_addr: bus[16];
  lane length: bus[32];
  lane desc_valid: bit;
  lane dma_done: bit;

  anchor first = rise(dma_start);
  anchor second = high(irq);
  anchor done = rise(desc_ready);

  window join_a = between first and done [1:1];
  window join_b = between second and done [1:1];

  show stable(status) from first until done;
  show stable(desc_addr) from first until done;
  show stable(length) from first until done;
  show low(desc_valid) before done;
  show low(dma_done) after done;
}
