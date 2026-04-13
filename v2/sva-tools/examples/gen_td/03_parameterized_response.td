diagram latency {
  clock posedge clk;
  disable iff !rst_n;

  param MIN_LAT;
  param MAX_LAT;

  lane grant: bit;
  lane req: bit;

  anchor latency__node_0 = rise(req);
  anchor latency__point_1 = 1'b1;
  anchor latency__node_2 = 1'b1;
  anchor latency__node_3 = rise(grant);

  window latency__win_0 = between latency__node_0 and latency__point_1 [0:0];
  window latency__win_1 = between latency__point_1 and latency__node_2 [0:0];
  window latency__win_2 = between latency__node_2 and latency__node_3 [MIN_LAT:MAX_LAT];

  cut latency__history_cut = before latency__node_0 omitted;
  cut latency__future_cut = after latency__node_3 omitted;

  property latency:
    $rose(req) |-> ##[MIN_LAT:MAX_LAT] $rose(grant);
}