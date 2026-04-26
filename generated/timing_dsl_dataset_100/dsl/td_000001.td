diagram td_chain_axi_like_td_000001 {
  clock posedge ACLK;
  ticks 11;

  param AW_READY_MAX;

  lane WSTRB: bus[16] = FF FF FF FF FF FF 0A x x x x;
  lane AWVALID: bit = 0 0 1 1 0 0 0 0 0 0 0;
  lane RREADY: bit = 0 0 0 1 0 0 0 0 0 0 0;
  lane AWREADY: bit = 0 0 0 0 0 0 1 0 0 0 0;
  lane BVALID: bit = 0 0 0 0 0 0 1 0 0 0 0;
  lane WDATA: bus[32] = x x 01 01 x x x 55 x x x;

  anchor trigger = rise(AWVALID);
  anchor middle = rise(RREADY);
  anchor response = high(AWREADY) and high(BVALID) and eq(WSTRB, 0A);

  window first_window = between trigger and middle [1:AW_READY_MAX];
  window second_window = between middle and response [3:3];

  cut cut_after_1 = after response compressed label "gap";

  show high(AWVALID) from trigger until middle;
  show stable(WDATA) in first_window;
  show low(AWREADY) before trigger;
}
