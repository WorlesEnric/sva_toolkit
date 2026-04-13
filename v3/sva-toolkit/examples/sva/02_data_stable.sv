property wait;
  @(posedge clk)
    $rose(valid) |-> ##[1:4] (valid && ready);
endproperty

property high_valid_from_start_until_done;
  @(posedge clk)
    $rose(valid) |-> valid until_with (valid && ready);
endproperty

property stable_data_from_start_until_done;
  @(posedge clk)
    $rose(valid) |-> $stable(data) until_with (valid && ready);
endproperty
