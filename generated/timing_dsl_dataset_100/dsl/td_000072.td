diagram td_parallel_axi_like_td_000072 {
  clock posedge ACLK;
  ticks 7;

  lane WDATA: bus[32] = 10 10 x x x x x;
  lane RDATA: bus[8] = 02 02 02 F0 x x x;
  lane AWVALID: bit = 0 1 0 0 0 0 0;
  lane RVALID: bit = 0 0 0 1 0 0 0;
  lane WREADY: bit = 0 0 0 1 0 0 0;
  lane WLAST: bit = 0 0 0 1 0 0 0;
  lane AWREADY: bit = 0 0 0 0 0 0 0;
  lane RLAST: bit = 0 0 0 0 0 0 0;
  lane WSTRB: bus[8] = x x 00 F0 x x x;

  anchor a_start = rise(AWVALID);
  anchor a_end = rise(RVALID);
  anchor b_start = stable(WDATA);
  anchor b_end = high(WREADY) and high(WLAST) and eq(RDATA, F0);

  window a_window = between a_start and a_end [1:2];
  window b_window = between b_start and b_end [2:2];

  show low(AWREADY) from a_start until a_end;
  show low(RLAST) before b_end;
  show WSTRB = F0 at a_end;
}
