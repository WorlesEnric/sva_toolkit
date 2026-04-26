diagram td_single_response_noc_td_000063 {
  clock negedge clk;
  ticks 17;

  param CR_MAX;

  lane TX_VALID: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RX_READY: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane RX_DATA: bus[8] = x x A1 A1 A1 A1 x x x x x x x 00 x x x;
  lane RX_VALID: bit = 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
  lane TX_READY: bit = 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;

  anchor trigger = rise(TX_VALID);
  anchor response = rise(RX_READY);

  window response_window = between trigger and response [1:CR_MAX];

  show stable(RX_DATA) in response_window;
  show low(RX_READY) before trigger;
}
