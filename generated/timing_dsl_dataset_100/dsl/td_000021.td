diagram td_backpressure_noc_td_000021 {
  clock negedge clk;
  ticks 10;

  lane RX_VALID: bit = 0 0 1 1 1 1 0 0 0 0;
  lane TX_VALID: bit = 0 0 0 0 0 1 0 0 0 0;
  lane CR_READY: bit = 0 0 0 0 0 1 0 0 0 0;
  lane TX_READY: bit = 0 0 0 0 0 0 0 0 0 0;
  lane TX_HDR: bus[8] = x x EE EE EE EE x x 7E x;
  lane RX_DATA: bus[8] = x x BE BE BE BE D0 x x x;
  lane CR_VALID: bit = 0 0 0 0 0 0 0 0 0 1;

  anchor valid_rise = rise(RX_VALID);
  anchor handshake = high(TX_VALID) and high(CR_READY);

  window wait_window = between valid_rise and handshake [1:3];

  cut cut_before_21 = before valid_rise omitted label "idle";

  show low(TX_VALID) before valid_rise;
  show high(RX_VALID) from valid_rise until handshake;
  show low(TX_READY) from valid_rise until handshake;
  show stable(TX_HDR) from valid_rise until handshake;
  show stable(RX_DATA) in wait_window;
}
