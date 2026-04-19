// Example 7: Symbolic-only diagram (no concrete samples)
// Demonstrates: pure symbolic lanes, multiple windows, constraint regions, lossy property

diagram symbolic_pipeline {
  clock negedge clk;
  disable iff !rst_n;

  param SETUP;
  param HOLD;
  param LATENCY;

  lane en:    bit;
  lane din:   bus[16];
  lane dout:  bus[16];
  lane valid: bit;
  lane stall: bit;

  anchor launch  = rise(en);
  anchor capture = rise(valid);
  anchor stalled = high(stall);

  window setup_window   = between launch and capture [SETUP:SETUP];
  window hold_window    = between capture and stalled [0:HOLD];
  window pipe_latency   = between launch and capture [1:LATENCY];

  cut pre_launch  = before launch omitted label "reset phase";
  cut post_capture = after capture omitted;

  show stable(din) from launch until capture;
  show low(stall) in setup_window;
  show high(valid) at capture;

  property setup_check:
    (en && !valid) |-> $stable(din) until_with valid;

  property latency_bound [lossy]:
    en |-> ##[1:LATENCY] valid;
  // latency_bound: actual latency depends on pipeline depth
}
