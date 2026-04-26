diagram td_parallel_fifo_td_000085 {
  clock posedge clk;
  ticks 10;

  lane wdata: bus[16] = F0 F0 0A x x x x x x x;
  lane level: bus[8] = 7E 7E 7E 7E FF x x x x x;
  lane rdata: bus[16] = 00 00 00 00 00 FF x x x x;
  lane pop: bit = 0 0 1 0 0 0 0 0 0 0;
  lane full: bit = 0 0 0 0 0 1 0 0 0 0;
  lane rreq: bit = 0 0 0 0 0 1 0 0 0 0;
  lane bus_0: bus[8] = x 33 EE EE EE x x x x x;

  anchor a_start = change(wdata);
  anchor a_end = change(level);
  anchor b_start = rise(pop);
  anchor b_end = high(full) and high(rreq) and eq(rdata, FF);

  window a_window = between a_start and a_end [1:3];
  window b_window = between b_start and b_end [2:3];

  show stable(bus_0) in a_window;
}
