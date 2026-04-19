property arb_phase;
  @(posedge clk)
    $rose(req) |-> ##[1:4] $rose(grant);
endproperty

property xfer_phase;
  @(posedge clk)
    $rose(grant) |-> ##[0:2] $rose(valid);
endproperty

property data_phase;
  @(posedge clk)
    $rose(valid) |-> ##[1:8] $fell(valid);
endproperty

property done_phase;
  @(posedge clk)
    $fell(valid) |-> ##[0:1] $rose(done);
endproperty

property high_req_from_req_assert_until_completion;
  @(posedge clk)
    $rose(req) |-> req until_with $rose(done);
endproperty

property stable_addr_from_req_assert_until_grant_assert;
  @(posedge clk)
    $rose(req) |-> $stable(addr) until_with $rose(grant);
endproperty

property high_grant_from_grant_assert_until_xfer_end;
  @(posedge clk)
    $rose(grant) |-> grant until_with $fell(valid);
endproperty

property high_valid_from_xfer_start_until_xfer_end;
  @(posedge clk)
    $rose(valid) |-> valid until_with $fell(valid);
endproperty

property low_err_from_req_assert_until_completion;
  @(posedge clk)
    $rose(req) |-> !err until_with $rose(done);
endproperty
