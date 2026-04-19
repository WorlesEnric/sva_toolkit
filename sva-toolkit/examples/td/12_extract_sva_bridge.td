diagram req_to_ack {
  clock posedge clk;
  disable iff !rst_n;

  lane ack: bit;
  lane req: bit;

  anchor req_to_ack__node_0 = high(req);
  anchor req_to_ack__point_1 = 1'b1;
  anchor req_to_ack__node_2 = 1'b1;
  anchor req_to_ack__node_3 = high(ack);

  window req_to_ack__win_0 = between req_to_ack__node_0 and req_to_ack__point_1 [0:0];
  window req_to_ack__win_1 = between req_to_ack__point_1 and req_to_ack__node_2 [0:0];
  window req_to_ack__win_2 = between req_to_ack__node_2 and req_to_ack__node_3 [1:3];

  cut req_to_ack__history_cut = before req_to_ack__node_0 omitted;
  cut req_to_ack__future_cut = after req_to_ack__node_3 omitted;

  property req_to_ack:
    req |-> ##[1:3] ack;
}