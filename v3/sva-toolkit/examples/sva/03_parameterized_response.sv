property latency(int MIN_LAT, int MAX_LAT);
  @(posedge clk) disable iff (!rst_n)
    $rose(req) |-> ##[MIN_LAT:MAX_LAT] $rose(grant);
endproperty
