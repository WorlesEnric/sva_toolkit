diagram data_latency__bundle_7 {
  clock posedge clk;
  disable iff !rst_n;

  param DEPTH;

  lane empty: bit;
  lane full: bit;
  lane pop: bit;
  lane push: bit;

  anchor data_latency__bundle_7__node_0 = rise(push) and low(full);
  anchor data_latency__bundle_7__point_1 = 1'b1;
  anchor data_latency__bundle_7__node_2 = 1'b1;
  anchor data_latency__bundle_7__node_3 = rise(pop) and low(empty);
  anchor data_latency__bundle_7__node_4 = rise(pop) and low(empty);
  anchor data_latency__bundle_7__node_5 = rise(empty);
  anchor data_latency__bundle_7__node_6 = rise(full);
  anchor data_latency__bundle_7__node_7 = low(empty);
  anchor data_latency__bundle_7__node_8 = low(full);

  window data_latency__bundle_7__win_0 = between data_latency__bundle_7__node_0 and data_latency__bundle_7__point_1 [0:0];
  window data_latency__bundle_7__win_1 = between data_latency__bundle_7__point_1 and data_latency__bundle_7__node_2 [0:0];
  window data_latency__bundle_7__win_2 = between data_latency__bundle_7__node_2 and data_latency__bundle_7__node_3 [1:$];
  window data_latency__bundle_7__win_3 = between data_latency__bundle_7__node_4 and data_latency__bundle_7__point_1 [0:0];
  window data_latency__bundle_7__win_4 = between data_latency__bundle_7__node_2 and data_latency__bundle_7__node_5 [1:DEPTH];
  window data_latency__bundle_7__win_5 = between data_latency__bundle_7__node_2 and data_latency__bundle_7__node_6 [1:DEPTH];
  window data_latency__bundle_7__win_6 = between data_latency__bundle_7__point_1 and data_latency__bundle_7__node_7 [0:0];
  window data_latency__bundle_7__win_7 = between data_latency__bundle_7__point_1 and data_latency__bundle_7__node_8 [0:0];
  window data_latency__bundle_7__win_8 = between data_latency__bundle_7__point_1 and data_latency__bundle_7__node_3 [0:0];
  window data_latency__bundle_7__win_9 = between data_latency__bundle_7__point_1 and data_latency__bundle_7__node_5 [0:0];
  window data_latency__bundle_7__win_10 = between data_latency__bundle_7__node_3 and data_latency__bundle_7__node_4 [0:0];
  window data_latency__bundle_7__win_11 = between data_latency__bundle_7__node_5 and data_latency__bundle_7__node_4 [0:0];
  window data_latency__bundle_7__win_12 = between data_latency__bundle_7__node_6 and data_latency__bundle_7__node_0 [0:0];
  window data_latency__bundle_7__win_13 = between data_latency__bundle_7__node_7 and data_latency__bundle_7__node_4 [0:0];
  window data_latency__bundle_7__win_14 = between data_latency__bundle_7__node_8 and data_latency__bundle_7__node_0 [0:0];

  property data_latency:
    $rose(push) && !full |-> ##[1:$] $rose(pop) && !empty;
  property drain_phase:
    $rose(pop) && !empty |-> ##[1:DEPTH] $rose(empty);
  property fill_phase:
    $rose(push) && !full |-> ##[1:DEPTH] $rose(full);
  property low_empty_at_first_pop:
    $rose(pop) && !empty |-> !empty;
  property low_full_at_first_push:
    $rose(push) && !full |-> !full;
  property low_pop_from_first_push_until_first_pop:
    $rose(push) && !full |-> !pop until_with $rose(pop) && !empty;
  property low_push_from_first_pop_until_fifo_empty:
    $rose(pop) && !empty |-> !push until_with $rose(empty);
}