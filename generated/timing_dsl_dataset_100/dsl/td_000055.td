diagram td_backpressure_axi_like_td_000055 {
  clock negedge ACLK;
  ticks 16;

  lane AWVALID: bit;
  lane AWREADY: bit;
  lane BREADY: bit;
  lane WSTRB: bus[32];
  lane BRESP: bus[32];
  lane WREADY: bit;
  lane RDATA: bus[32];
  lane WLAST: bit;
  lane RVALID: bit;

  anchor valid_rise = rise(AWVALID);
  anchor handshake = high(AWREADY) and high(BREADY);

  window wait_window = between valid_rise and handshake [2:5];

  cut cut_after_55 = after handshake compressed label "after region intentionally hidden for timing context";

  show stable(WSTRB) from valid_rise until handshake;
  show stable(BRESP) from valid_rise until handshake;
  show high(AWVALID) from valid_rise until handshake;
  show low(WREADY) from valid_rise until handshake;
  show stable(RDATA) from valid_rise until handshake;
}
