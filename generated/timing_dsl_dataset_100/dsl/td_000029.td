diagram td_chain_dma_td_000029 {
  clock posedge clk;
  ticks 11;

  lane dma_start: bit = 0 0 1 0 0 0 0 0 0 0 0;
  lane desc_valid: bit = 0 0 0 0 1 1 0 0 0 0 0;
  lane desc_ready: bit = 0 0 0 0 0 1 0 0 0 0 0;
  lane irq: bit = 0 0 0 0 0 1 0 0 0 0 0;
  lane dma_done: bit = 0 0 0 0 0 0 0 0 0 0 0;
  lane status: bus[32] = A5 A1 x x x x x x x D0 x;
  lane bit_0: bit = 0 0 0 0 0 0 0 0 1 0 0;

  anchor trigger = rise(dma_start);
  anchor middle = rise(desc_valid);
  anchor response = high(desc_ready) and high(irq);

  window first_window = between trigger and middle [2:2];
  window second_window = between middle and response [1:1];

  show low(dma_done) from trigger until middle;
  show high(desc_valid) from middle until response;
  show neq(status, F0) before trigger;
  show low(bit_0) at middle;
}
