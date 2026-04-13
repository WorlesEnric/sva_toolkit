property ack_after_req__window;
  @(posedge clk)
    $rose(req) |-> ##[1:5] $rose(ack);
endproperty

property high_req_from_req_start_until_ack_seen;
  @(posedge clk)
    $rose(req) |-> req until_with $rose(ack);
endproperty
