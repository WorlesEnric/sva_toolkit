property response_time;
  @(posedge clk)
    $rose(req) |-> ##[1:4] $rose(ack);
endproperty
