diagram td_backpressure_generic_td_000076 {
  clock negedge clk;
  ticks 20;

  lane busy: bit;
  lane resp_valid: bit;
  lane valid: bit;
  lane ready: bit;
  lane tag: bus[8];

  anchor valid_rise = rise(busy);
  anchor handshake = high(resp_valid) and high(valid);

  window wait_window = between valid_rise and handshake [2:2];

  show high(busy) from valid_rise until handshake;
  show high(busy) from valid_rise until handshake;
  show low(ready) from valid_rise until handshake;
  show stable(tag) from valid_rise until handshake;
}
