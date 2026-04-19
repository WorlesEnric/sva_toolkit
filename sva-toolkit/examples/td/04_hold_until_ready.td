// Example 4: Hold-until with cuts
// Demonstrates: cuts (omitted history/future), hold-style constraint, property

diagram hold_until_ready {
  clock posedge clk;
  ticks 8;

  lane valid: bit = 0 1 1 1 1 1 0 0;
  lane ready: bit = 0 0 0 0 0 1 0 0;
  lane data:  bus[32] = x D0 D0 D0 D0 D0 x x;

  anchor asserted   = rise(valid);
  anchor handshake  = high(valid) and high(ready);

  window hold_period = between asserted and handshake [0:8];

  cut history = before asserted omitted;
  cut future  = after handshake omitted;

  show high(valid) from asserted until handshake;
  show stable(data) from asserted until handshake;

  property valid_hold:
    (valid && !ready) |-> ($stable(data) && valid) until_with ready;
}
