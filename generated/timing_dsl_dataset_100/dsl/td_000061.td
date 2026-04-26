diagram td_chain_noc_td_000061 {
  clock negedge clk;
  ticks 20;

  param CR_MAX;

  lane CR_VALID: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_VALID: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CR_READY: bit = 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RX_DATA: bus[32] = x 10 10 10 x x x x x x x x 7E x x x x x x x;
  lane TX_DATA: bus[8] = x x x 10 10 10 10 x x x x x x x x BE x x x x;
  lane VC: bus[32] = x x x 01 01 01 01 x x x x x x x x x x 55 x x;
  lane RX_VALID: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = rise(CR_VALID);
  anchor middle = high(TX_VALID);
  anchor response = rise(CR_READY);

  window first_window = between trigger and middle [1:3];
  window second_window = between middle and response [1:CR_MAX];

  show stable(RX_DATA) from trigger until middle;
  show stable(TX_DATA) from middle until response;
  show stable(VC) from middle until response;
  show low(CR_READY) before trigger;
  show high(RX_VALID) at middle;
}
