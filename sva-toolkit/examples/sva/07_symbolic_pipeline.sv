property setup_window(int SETUP);
  @(negedge clk) disable iff (!rst_n)
    $rose(en) |-> ##SETUP $rose(valid);
endproperty

property hold_window(int HOLD);
  @(negedge clk) disable iff (!rst_n)
    $rose(valid) |-> ##[0:HOLD] stall;
endproperty

property pipe_latency(int LATENCY);
  @(negedge clk) disable iff (!rst_n)
    $rose(en) |-> ##[1:LATENCY] $rose(valid);
endproperty

property stable_din_from_launch_until_capture;
  @(negedge clk) disable iff (!rst_n)
    $rose(en) |-> $stable(din) until_with $rose(valid);
endproperty

property high_valid_at_capture;
  @(negedge clk) disable iff (!rst_n)
    $rose(valid) |-> valid;
endproperty
