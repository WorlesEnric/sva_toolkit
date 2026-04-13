diagram response_time {
  clock posedge clk;
  ticks 3;

  lane ack: bit = 0 0 1;
  lane req: bit = 1 0 0;

  anchor response_time__node_0 = rise(req);
  anchor response_time__point_1 = 1'b1;
  anchor response_time__node_2 = 1'b1;
  anchor response_time__node_3 = rise(ack);

  window response_time__win_0 = between response_time__node_0 and response_time__point_1 [0:0];
  window response_time__win_1 = between response_time__point_1 and response_time__node_2 [0:0];
  window response_time__win_2 = between response_time__node_2 and response_time__node_3 [1:4];

  cut response_time__history_cut = before response_time__node_0 omitted;
  cut response_time__future_cut = after response_time__node_3 omitted;

  property response_time:
    $rose(req) |-> ##[1:4] $rose(ack);
}