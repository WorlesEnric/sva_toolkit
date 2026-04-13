diagram high_valid_from_start_until_done__bundle_3 {
  clock posedge clk;
  ticks 49;

  lane ready: bit = 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
  lane valid: bit = 1 1 0 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 1 0 1 1 0 1 0 1;
  lane data: bus = x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x;

  anchor high_valid_from_start_until_done__bundle_3__node_0 = rise(valid);
  anchor high_valid_from_start_until_done__bundle_3__point_1 = 1'b1;
  anchor high_valid_from_start_until_done__bundle_3__node_2 = high(valid) and high(ready);
  anchor high_valid_from_start_until_done__bundle_3__node_3 = 1'b1;

  window high_valid_from_start_until_done__bundle_3__win_0 = between high_valid_from_start_until_done__bundle_3__node_0 and high_valid_from_start_until_done__bundle_3__point_1 [0:0];
  window high_valid_from_start_until_done__bundle_3__win_1 = between high_valid_from_start_until_done__bundle_3__point_1 and high_valid_from_start_until_done__bundle_3__node_2 [0:0];
  window high_valid_from_start_until_done__bundle_3__win_2 = between high_valid_from_start_until_done__bundle_3__point_1 and high_valid_from_start_until_done__bundle_3__node_3 [0:0];
  window high_valid_from_start_until_done__bundle_3__win_3 = between high_valid_from_start_until_done__bundle_3__node_3 and high_valid_from_start_until_done__bundle_3__node_2 [1:4];
  window high_valid_from_start_until_done__bundle_3__win_4 = between high_valid_from_start_until_done__bundle_3__node_2 and high_valid_from_start_until_done__bundle_3__node_0 [0:0];

  property high_valid_from_start_until_done:
    $rose(valid) |-> valid until_with valid && ready;
  property stable_data_from_start_until_done:
    $rose(valid) |-> $stable(data) until_with valid && ready;
  property wait:
    $rose(valid) |-> ##[1:4] valid && ready;
}