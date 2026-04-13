diagram response_window {
  clock posedge clk;
  disable iff !rst_n;
  ticks 2;

  param MAX_LATENCY;

  lane ack: bit = x 1;
  lane req: bit = 1 0;

  anchor response_window__node_0 = high(req);
  anchor response_window__point_1 = 1'b1;
  anchor response_window__node_2 = 1'b1;
  anchor response_window__node_3 = high(ack);

  window response_window__win_0 = between response_window__node_0 and response_window__point_1 [0:0];
  window response_window__win_1 = between response_window__point_1 and response_window__node_2 [0:0];
  window response_window__win_2 = between response_window__node_2 and response_window__node_3 [1:MAX_LATENCY];

  cut response_window__history_cut = before response_window__node_0 omitted;
  cut response_window__future_cut = after response_window__node_3 omitted;

  property response_window:
    req |-> ##[1:MAX_LATENCY] ack;
}