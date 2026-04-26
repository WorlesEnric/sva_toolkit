diagram td_setup_hold_axi_like_td_000044 {
  clock negedge ACLK;
  ticks 14;

  param AW_READY_MAX;

  lane AWVALID: bit = 0 0 1 1 1 1 0 0 0 0 0 0 0 0;
  lane ARREADY: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane RVALID: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RLAST: bit = 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
  lane BVALID: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor capture = high(AWVALID);
  anchor hold_end = high(ARREADY);

  window hold_window = between capture and hold_end [1:AW_READY_MAX];

  show high(AWVALID) from capture until hold_end;
  show low(RVALID) at hold_end;
}
