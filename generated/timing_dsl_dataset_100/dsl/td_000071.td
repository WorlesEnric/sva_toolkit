diagram td_burst_fifo_td_000071 {
  clock posedge clk;
  ticks 12;

  lane level: bus[16] = FF FF FF FF FF FF FF D0 x x x x;
  lane push: bit = 0 0 1 0 0 0 0 0 0 0 0 0;
  lane rreq: bit = 0 0 1 0 0 0 0 0 0 0 0 0;
  lane empty: bit = 0 0 0 0 1 0 0 0 0 0 0 0;
  lane full: bit = 0 0 0 0 1 0 0 0 0 0 0 0;
  lane pop: bit = 0 0 0 0 0 0 0 1 0 0 0 0;
  lane wreq: bit = 0 0 0 0 0 0 0 1 0 0 0 0;
  lane bit_0: bit = 0 0 0 0 0 0 0 0 1 0 0 0;
  lane wdata: bus[16] = 0A 02 x D0 x x x x x x x x;
  lane rdata: bus[8] = x 00 D0 BE A5 A1 10 0A x x x x;
  lane bit_1: bit = 0 0 0 0 0 0 0 0 0 0 1 0;

  anchor first_beat = high(push) and high(rreq);
  anchor middle_beat = high(empty) and high(full);
  anchor last_beat = high(pop) and high(wreq) and eq(level, D0);

  window burst_first_to_middle = between first_beat and middle_beat [2:4];
  window burst_window = between middle_beat and last_beat [2:4];

  show low(bit_0) from first_beat until middle_beat;
  show neq(wdata, A5) before first_beat;
  show change(rdata) at first_beat;
  show change(rdata) at middle_beat;
  show change(rdata) at last_beat;
  show low(bit_1) at last_beat;
}
