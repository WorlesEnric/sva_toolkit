property response_window(int MAX_LATENCY);
  @(posedge clk) disable iff (!rst_n)
    req |-> ##[1:MAX_LATENCY] ack;
endproperty
