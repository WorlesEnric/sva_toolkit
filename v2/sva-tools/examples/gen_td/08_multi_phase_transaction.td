diagram arb_phase__bundle_9 {
  clock posedge clk;

  lane grant: bit;
  lane req: bit;
  lane valid: bit;
  lane done: bit;
  lane err: bit;
  lane addr: bus;

  anchor arb_phase__bundle_9__node_0 = rise(req);
  anchor arb_phase__bundle_9__point_1 = 1'b1;
  anchor arb_phase__bundle_9__node_2 = 1'b1;
  anchor arb_phase__bundle_9__node_3 = rise(grant);
  anchor arb_phase__bundle_9__node_4 = rise(valid);
  anchor arb_phase__bundle_9__node_5 = fall(valid);
  anchor arb_phase__bundle_9__node_6 = fall(valid);
  anchor arb_phase__bundle_9__node_7 = rise(done);
  anchor arb_phase__bundle_9__node_8 = rise(grant);
  anchor arb_phase__bundle_9__node_9 = rise(valid);

  window arb_phase__bundle_9__win_0 = between arb_phase__bundle_9__node_0 and arb_phase__bundle_9__point_1 [0:0];
  window arb_phase__bundle_9__win_1 = between arb_phase__bundle_9__point_1 and arb_phase__bundle_9__node_2 [0:0];
  window arb_phase__bundle_9__win_2 = between arb_phase__bundle_9__node_2 and arb_phase__bundle_9__node_3 [1:4];
  window arb_phase__bundle_9__win_3 = between arb_phase__bundle_9__node_4 and arb_phase__bundle_9__point_1 [0:0];
  window arb_phase__bundle_9__win_4 = between arb_phase__bundle_9__node_2 and arb_phase__bundle_9__node_5 [1:8];
  window arb_phase__bundle_9__win_5 = between arb_phase__bundle_9__node_6 and arb_phase__bundle_9__point_1 [0:0];
  window arb_phase__bundle_9__win_6 = between arb_phase__bundle_9__node_2 and arb_phase__bundle_9__node_7 [0:1];
  window arb_phase__bundle_9__win_7 = between arb_phase__bundle_9__node_8 and arb_phase__bundle_9__point_1 [0:0];
  window arb_phase__bundle_9__win_8 = between arb_phase__bundle_9__point_1 and arb_phase__bundle_9__node_5 [0:0];
  window arb_phase__bundle_9__win_9 = between arb_phase__bundle_9__point_1 and arb_phase__bundle_9__node_7 [0:0];
  window arb_phase__bundle_9__win_10 = between arb_phase__bundle_9__point_1 and arb_phase__bundle_9__node_3 [0:0];
  window arb_phase__bundle_9__win_11 = between arb_phase__bundle_9__node_2 and arb_phase__bundle_9__node_9 [0:2];
  window arb_phase__bundle_9__win_12 = between arb_phase__bundle_9__node_3 and arb_phase__bundle_9__node_8 [0:0];
  window arb_phase__bundle_9__win_13 = between arb_phase__bundle_9__node_5 and arb_phase__bundle_9__node_6 [0:0];
  window arb_phase__bundle_9__win_14 = between arb_phase__bundle_9__node_5 and arb_phase__bundle_9__node_4 [0:0];

  property arb_phase:
    $rose(req) |-> ##[1:4] $rose(grant);
  property data_phase:
    $rose(valid) |-> ##[1:8] $fell(valid);
  property done_phase:
    $fell(valid) |-> ##[0:1] $rose(done);
  property high_grant_from_grant_assert_until_xfer_end:
    $rose(grant) |-> grant until_with $fell(valid);
  property high_req_from_req_assert_until_completion:
    $rose(req) |-> req until_with $rose(done);
  property high_valid_from_xfer_start_until_xfer_end:
    $rose(valid) |-> valid until_with $fell(valid);
  property low_err_from_req_assert_until_completion:
    $rose(req) |-> !err until_with $rose(done);
  property stable_addr_from_req_assert_until_grant_assert:
    $rose(req) |-> $stable(addr) until_with $rose(grant);
  property xfer_phase:
    $rose(grant) |-> ##[0:2] $rose(valid);
}