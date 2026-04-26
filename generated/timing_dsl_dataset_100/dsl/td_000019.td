diagram td_chain_dma_td_000019 {
  clock posedge clk;
  ticks 14;

  lane desc_addr: bus[16] = D0 D0 D0 D0 D0 D0 33 x x x x x x x;
  lane desc_ready: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane irq: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane desc_valid: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane dma_start: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
  lane status: bus[32] = x EE EE EE x 10 x x x x x x x x;
  lane length: bus[16] = x 33 33 33 x x x x x x x x x EE;
  lane bus_0: bus[32] = x FF FF FF x x 1F x x x x x x x;
  lane bus_1: bus[16] = x 7E x x x x 10 10 10 10 10 10 10 10;

  anchor trigger = rise(desc_ready);
  anchor middle = rise(irq);
  anchor response = high(desc_valid) and high(dma_start) and eq(desc_addr, 33);

  window first_window = between trigger and middle [1:2];
  window second_window = between middle and response [2:4];

  show stable(status) from trigger until middle;
  show stable(length) from trigger until middle;
  show bus_0 = FF in first_window;
  show bus_1 = 10 after response;
}
