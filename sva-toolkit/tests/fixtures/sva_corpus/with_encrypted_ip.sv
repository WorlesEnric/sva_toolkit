`protect begin_protected
assert property (@(posedge clk) disable iff (!rst_n) secret_req |-> ##1 secret_ack);
`endprotect
