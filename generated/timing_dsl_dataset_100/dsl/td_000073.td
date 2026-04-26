diagram td_setup_hold_dma_td_000073 {
  clock negedge clk;
  ticks 16;

  lane desc_valid: bit = 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane dma_done: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane dma_start: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane desc_addr: bus[16] = x x BE BE x x x x x x x x x D0 x x;
  lane desc_ready: bit = 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane length: bus[32] = x x EE EE x x D0 x x x x x x x x x;
  lane status: bus[32] = 1F x x x x x 7E 7E 7E 7E 7E 7E 7E 7E 7E 7E;
  lane irq: bit;

  anchor launch = rise(desc_valid);
  anchor capture = high(dma_done);
  anchor hold_end = high(dma_start);

  window setup_window = between launch and capture [1:3];
  window hold_window = between capture and hold_end [2:3];

  show high(desc_valid) from launch until capture;
  show stable(desc_addr) in setup_window;
  show low(dma_start) before launch;
  show high(desc_ready) at capture;
  show length = EE in setup_window;
  show status = 7E after hold_end;
}
