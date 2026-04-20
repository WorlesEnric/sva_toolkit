(* severity = "error", owner = "integration" *)
assert property (@(posedge clk) disable iff (!rst_n) req |-> ##1 ack);
