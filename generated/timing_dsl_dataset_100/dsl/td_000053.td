diagram td_burst_axi_like_td_000053 {
  clock posedge ACLK;
  ticks 17;

  lane ARADDR: bus[32] = 01 01 01 01 D0 x x x x x x x x x x x x;
  lane AWVALID: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane ARREADY: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane AWREADY: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane WLAST: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane BVALID: bit = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane WSTRB: bus[32] = 01 x 7E 7E 7E x x x x x x x x x x x x;
  lane WDATA: bus[8] = x 01 A5 01 02 x x x x x x x x x x F0 x;
  lane AWADDR: bus[32] = 7E x 33 33 33 x x x x x x x x x x x x;
  lane BRESP: bus[8] = x x x 1F x x x x A1 A1 A1 A1 A1 A1 A1 A1 A1;

  anchor first_beat = high(AWVALID) and high(ARREADY);
  anchor last_beat = high(AWREADY) and high(WLAST) and eq(ARADDR, D0);
  anchor response_after_last = rise(BVALID);

  window burst_window = between first_beat and last_beat [2:4];
  window response_after_last_window = between last_beat and response_after_last [2:5];

  show stable(WSTRB) in burst_window;
  show low(AWREADY) before first_beat;
  show change(WDATA) at first_beat;
  show change(WDATA) at last_beat;
  show stable(AWADDR) in burst_window;
  show BRESP = A1 after response_after_last;
}
