diagram td_single_response_fifo_td_000090 {
  clock posedge clk;
  ticks 15;

  lane rreq: bit = 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane empty: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane wreq: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane full: bit = 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0;
  lane wdata: bus[8] = x A1 A1 A1 A1 A1 x x x x EE x x x x;
  lane level: bus[8] = x A5 A5 A5 A5 A5 x x x x 0A x x x x;
  lane rdata: bus[8] = 01 33 33 33 33 33 x x x x x x x x x;
  lane bus_0: bus[8] = x x A5 x x 02 0A 0A 33 FF F0 7E 01 55 7E;

  anchor trigger = low(rreq);
  anchor response = high(empty) and high(wreq);

  window response_window = between trigger and response [2:4];

  show high(full) from trigger until response;
  show stable(wdata) from trigger until response;
  show stable(level) from trigger until response;
  show low(empty) before trigger;
  show stable(rdata) in response_window;
  show neq(bus_0, A5) after response;
}
