property hold_period;
  @(posedge clk)
    $rose(valid) |-> ##[0:8] (valid && ready);
endproperty

property high_valid_from_asserted_until_handshake;
  @(posedge clk)
    $rose(valid) |-> valid until_with (valid && ready);
endproperty

property stable_data_from_asserted_until_handshake;
  @(posedge clk)
    $rose(valid) |-> $stable(data) until_with (valid && ready);
endproperty
