assert property (@(posedge clk) disable iff (!rst_n) mode == "RUN" |-> ##1 ack);
