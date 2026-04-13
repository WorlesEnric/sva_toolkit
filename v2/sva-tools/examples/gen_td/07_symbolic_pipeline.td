diagram high_valid_at_capture__bundle_5 {
  clock negedge clk;
  disable iff !rst_n;

  param HOLD;
  param LATENCY;
  param SETUP;

  lane valid: bit;
  lane stall: bit;
  lane en: bit;
  lane din: bus;

  anchor high_valid_at_capture__bundle_5__node_0 = rise(valid);
  anchor high_valid_at_capture__bundle_5__point_1 = 1'b1;
  anchor high_valid_at_capture__bundle_5__node_2 = high(valid);
  anchor high_valid_at_capture__bundle_5__node_3 = 1'b1;
  anchor high_valid_at_capture__bundle_5__node_4 = high(stall);
  anchor high_valid_at_capture__bundle_5__node_5 = rise(en);
  anchor high_valid_at_capture__bundle_5__node_6 = rise(valid);

  window high_valid_at_capture__bundle_5__win_0 = between high_valid_at_capture__bundle_5__node_0 and high_valid_at_capture__bundle_5__point_1 [0:0];
  window high_valid_at_capture__bundle_5__win_1 = between high_valid_at_capture__bundle_5__point_1 and high_valid_at_capture__bundle_5__node_2 [0:0];
  window high_valid_at_capture__bundle_5__win_2 = between high_valid_at_capture__bundle_5__point_1 and high_valid_at_capture__bundle_5__node_3 [0:0];
  window high_valid_at_capture__bundle_5__win_3 = between high_valid_at_capture__bundle_5__node_3 and high_valid_at_capture__bundle_5__node_4 [0:HOLD];
  window high_valid_at_capture__bundle_5__win_4 = between high_valid_at_capture__bundle_5__node_5 and high_valid_at_capture__bundle_5__point_1 [0:0];
  window high_valid_at_capture__bundle_5__win_5 = between high_valid_at_capture__bundle_5__node_3 and high_valid_at_capture__bundle_5__node_6 [1:LATENCY];
  window high_valid_at_capture__bundle_5__win_6 = between high_valid_at_capture__bundle_5__node_3 and high_valid_at_capture__bundle_5__node_6 [SETUP:SETUP];
  window high_valid_at_capture__bundle_5__win_7 = between high_valid_at_capture__bundle_5__point_1 and high_valid_at_capture__bundle_5__node_6 [0:0];
  window high_valid_at_capture__bundle_5__win_8 = between high_valid_at_capture__bundle_5__node_2 and high_valid_at_capture__bundle_5__node_0 [0:0];

  property high_valid_at_capture:
    $rose(valid) |-> valid;
  property hold_window:
    $rose(valid) |-> ##[0:HOLD] stall;
  property pipe_latency:
    $rose(en) |-> ##[1:LATENCY] $rose(valid);
  property setup_window:
    $rose(en) |-> ##SETUP $rose(valid);
  property stable_din_from_launch_until_capture:
    $rose(en) |-> $stable(din) until_with $rose(valid);
}