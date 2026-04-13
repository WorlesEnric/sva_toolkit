diagram ack_after_req__window__bundle_2 {
  clock posedge clk;
  ticks 81;

  lane ack: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
  lane req: bit = 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;

  anchor ack_after_req__window__bundle_2__node_0 = rise(req);
  anchor ack_after_req__window__bundle_2__point_1 = 1'b1;
  anchor ack_after_req__window__bundle_2__node_2 = 1'b1;
  anchor ack_after_req__window__bundle_2__node_3 = rise(ack);

  window ack_after_req__window__bundle_2__win_0 = between ack_after_req__window__bundle_2__node_0 and ack_after_req__window__bundle_2__point_1 [0:0];
  window ack_after_req__window__bundle_2__win_1 = between ack_after_req__window__bundle_2__point_1 and ack_after_req__window__bundle_2__node_2 [0:0];
  window ack_after_req__window__bundle_2__win_2 = between ack_after_req__window__bundle_2__node_2 and ack_after_req__window__bundle_2__node_3 [1:5];
  window ack_after_req__window__bundle_2__win_3 = between ack_after_req__window__bundle_2__point_1 and ack_after_req__window__bundle_2__node_3 [0:0];

  property ack_after_req__window:
    $rose(req) |-> ##[1:5] $rose(ack);
  property high_req_from_req_start_until_ack_seen:
    $rose(req) |-> req until_with $rose(ack);
}