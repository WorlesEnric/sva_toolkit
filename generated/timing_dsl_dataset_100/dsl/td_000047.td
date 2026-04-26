diagram td_burst_fifo_td_000047 {
  clock posedge clk;
  ticks 14;

  param FIFO_MAX;

  lane level: bus[8] = BE BE BE A1 x x x x x x x x x x;
  lane push: bit = 0 0 1 1 0 0 0 0 0 0 0 0 0 0;
  lane empty: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane pop: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane full: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane wreq: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane rreq: bit = 0 0 0 1 1 1 0 0 0 0 0 0 0 0;
  lane rdata: bus[16] = x x 1F 1F x x x x x x x x BE x;
  lane wdata: bus[16] = 55 D0 x x x x x x A5 x x x x x;
  lane bus_0: bus[8] = x 7E BE D0 x 1F x x x x x x x x;
  lane bus_1: bus[8] = x x x x x 7E x x x x x x 33 x;

  anchor first_beat = high(push) and high(empty);
  anchor last_beat = high(pop) and high(full) and eq(level, A1);
  anchor response_after_last = rise(wreq);

  window burst_window = between first_beat and last_beat [1:FIFO_MAX];
  window response_after_last_window = between last_beat and response_after_last [2:3];

  show high(push) from first_beat until last_beat;
  show stable(rreq) from last_beat until response_after_last;
  show stable(rdata) from first_beat until last_beat;
  show neq(wdata, 7E) before first_beat;
  show change(bus_0) at first_beat;
  show change(bus_0) at last_beat;
  show neq(bus_1, D0) at response_after_last;
}
