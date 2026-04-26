diagram td_backpressure_axi_like_td_000017 {
  clock negedge ACLK;
  ticks 17;

  param AW_READY_MAX;

  lane AWVALID: bit = 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane AWREADY: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RVALID: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RREADY: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane WDATA: bus[8] = x 02 02 02 02 x 33 x x x x x x x x x x;
  lane WREADY: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
  lane RRESP: bus[8] = x 0A 0A 0A 0A x x x x x x x x x A5 x x;
  lane AWADDR: bus[32] = x 1F F0 A1 02 x x x x x x x 0A x x x x;

  anchor valid_rise = rise(AWVALID);
  anchor handshake = high(AWREADY) and high(RVALID);

  window wait_window = between valid_rise and handshake [1:AW_READY_MAX];

  show low(RREADY) from valid_rise until handshake;
  show stable(WDATA) from valid_rise until handshake;
  show low(AWREADY) before valid_rise;
  show high(AWVALID) from valid_rise until handshake;
  show low(WREADY) from valid_rise until handshake;
  show stable(RRESP) from valid_rise until handshake;
  show neq(AWADDR, 55) in wait_window;
}
