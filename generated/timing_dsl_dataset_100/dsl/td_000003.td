diagram td_join_noc_td_000003 {
  clock posedge clk;
  ticks 11;

  param CR_MAX;
  param TX_MAX;

  lane TX_DATA: bus[8] = A1 A1 1F x x x x x x x x;
  lane TX_READY: bit = 0 0 1 0 0 0 0 0 0 0 0;
  lane RX_READY: bit = 0 0 0 0 0 1 0 0 0 0 0;
  lane RX_DATA: bus[16] = x x BE BE BE BE x 55 x x x;
  lane TX_HDR: bus[32] = x x BE BE BE BE 7E x x x x;
  lane VC: bus[32] = 01 x F0 F0 F0 F0 x x x x x;
  lane bus_0: bus[32] = x x F0 F0 F0 F0 02 x x x x;
  lane TX_VALID: bit = 0 0 0 0 1 0 0 0 0 0 0;

  anchor first = rise(TX_READY);
  anchor second = eq(TX_DATA, 1F);
  anchor done = rise(RX_READY);

  window join_a = between first and done [1:CR_MAX];
  window join_b = between second and done [1:TX_MAX];

  cut cut_before_3 = before first omitted label "before region intentionally hidden for timing context";

  show stable(RX_DATA) from first until done;
  show stable(TX_HDR) from first until done;
  show VC = F0 from second until done;
  show stable(bus_0) from first until done;
  show fall(TX_VALID) at done;
}
