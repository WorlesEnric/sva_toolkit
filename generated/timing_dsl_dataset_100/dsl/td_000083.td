diagram td_join_axi_like_td_000083 {
  clock negedge ACLK;
  ticks 9;

  lane BRESP: bus[32] = BE 7E 7E x x x x x x;
  lane AWADDR: bus[8] = 55 F0 55 F0 F0 x x x x;
  lane BVALID: bit = 0 0 1 0 0 0 0 0 0;
  lane RRESP: bus[16] = x FF 10 10 10 x x x x;
  lane RDATA: bus[16] = x x 1F 1F 1F 7E x x x;

  anchor first = rise(BVALID);
  anchor second = stable(BRESP);
  anchor done = stable(AWADDR);

  window join_a = between first and done [1:3];
  window join_b = between second and done [1:3];

  cut cut_between_83 = between join_a and join_b lookback label "between region intentionally hidden for timing context";

  show stable(RRESP) in join_a;
  show RDATA = 1F in join_a;
}
