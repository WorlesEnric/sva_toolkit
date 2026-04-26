diagram td_single_response_noc_td_000033 {
  clock negedge clk;
  ticks 6;

  param CR_MAX;

  lane RX_READY: bit = 0 1 0 0 0 0;
  lane TX_VALID: bit = 0 0 1 0 0 0;
  lane RX_DATA: bus[32] = x 1F 1F x 10 x;
  lane TX_HDR: bus[8] = x FF FF 33 x x;
  lane TX_DATA: bus[32] = 00 F0 F0 x x x;
  lane VC: bus[8] = x A1 A1 A5 x x;
  lane bus_0: bus[32] = x F0 02 02 02 02;

  anchor trigger = rise(RX_READY);
  anchor response = rise(TX_VALID);

  window response_window = between trigger and response [1:CR_MAX];

  show stable(RX_DATA) from trigger until response;
  show stable(TX_HDR) from trigger until response;
  show stable(TX_DATA) in response_window;
  show VC = A1 in response_window;
  show bus_0 = 02 after response;
}
