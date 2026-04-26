diagram td_single_response_dma_td_000035 {
  clock posedge clk;
  ticks 20;

  lane desc_addr: bus[32] = FF FF F0 x x x x x x x x x x x x x x x x x;
  lane dma_start: bit = 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane dma_done: bit = 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane desc_valid: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
  lane status: bus[32] = x x FF FF x x x x x x x x x x D0 x x x x x;
  lane length: bus[16] = 01 x x A1 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1 A1;

  anchor trigger = eq(desc_addr, F0);
  anchor response = fall(dma_start);

  window response_window = between trigger and response [1:$];

  show high(dma_done) from trigger until response;
  show low(desc_valid) before response;
  show status = FF in response_window;
  show stable(length) after response;
}
