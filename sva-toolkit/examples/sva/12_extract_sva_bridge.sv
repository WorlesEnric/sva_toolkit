property req_to_ack__win_2;
  @(posedge clk) disable iff (!rst_n)
    req |-> ##[1:3] ack;
endproperty
