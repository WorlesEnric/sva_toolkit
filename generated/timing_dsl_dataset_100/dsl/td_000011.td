diagram td_single_response_dma_td_000011 {
  clock negedge clk;
  ticks 9;

  lane dma_start: bit = 0 1 0 0 0 0 0 0 0;
  lane desc_ready: bit = 0 0 0 0 0 1 0 0 0;
  lane irq: bit = 0 0 0 0 0 1 0 0 0;
  lane dma_done: bit = 0 0 0 0 0 0 0 0 0;
  lane status: bus[32] = x 10 10 10 10 10 A5 x x;

  anchor trigger = rise(dma_start);
  anchor response = high(desc_ready) and high(irq);

  window response_window = between trigger and response [2:4];

  show low(dma_done) from trigger until response;
  show low(desc_ready) before trigger;
  show status = 10 in response_window;
}
