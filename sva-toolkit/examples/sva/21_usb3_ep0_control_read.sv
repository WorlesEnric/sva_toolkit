property setup_wait(int SETUP_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(SETUP_VALID) |-> ##[0:SETUP_READY_MAX] (SETUP_VALID && SETUP_READY);
endproperty

property data_start(int DATA_START_MAX);
  @(posedge clk) disable iff (!rst_n)
    (SETUP_VALID && SETUP_READY) |-> ##[1:DATA_START_MAX] $rose(DATA_VALID);
endproperty

property data_first_gap(int DATA_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(DATA_VALID) |-> ##[0:DATA_READY_MAX] (DATA_VALID && DATA_READY);
endproperty

property data_phase;
  @(posedge clk) disable iff (!rst_n)
    $rose(DATA_VALID) |-> ##[0:8] (DATA_VALID && DATA_READY && DATA_LAST);
endproperty

property status_start;
  @(posedge clk) disable iff (!rst_n)
    (DATA_VALID && DATA_READY && DATA_LAST) |-> ##[1:2] $rose(STATUS_VALID);
endproperty

property status_wait(int STATUS_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(STATUS_VALID) |-> ##[0:STATUS_READY_MAX] (STATUS_VALID && STATUS_READY);
endproperty

property high_SETUP_VALID_from_setup_rise_until_setup_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(SETUP_VALID) |-> SETUP_VALID until_with (SETUP_VALID && SETUP_READY);
endproperty

property stable_BMREQ_from_setup_rise_until_setup_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(SETUP_VALID) |-> $stable(BMREQ) until_with (SETUP_VALID && SETUP_READY);
endproperty

property stable_BREQ_from_setup_rise_until_setup_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(SETUP_VALID) |-> $stable(BREQ) until_with (SETUP_VALID && SETUP_READY);
endproperty

property stable_WVALUE_from_setup_rise_until_setup_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(SETUP_VALID) |-> $stable(WVALUE) until_with (SETUP_VALID && SETUP_READY);
endproperty

property stable_WLEN_from_setup_rise_until_setup_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(SETUP_VALID) |-> $stable(WLEN) until_with (SETUP_VALID && SETUP_READY);
endproperty

property high_DATA_VALID_from_data_rise_until_data_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(DATA_VALID) |-> DATA_VALID until_with (DATA_VALID && DATA_READY && DATA_LAST);
endproperty

property stable_DATA_PID_from_data_rise_until_data_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(DATA_VALID) |-> $stable(DATA_PID) until_with (DATA_VALID && DATA_READY && DATA_LAST);
endproperty

property stable_DATA_BYTES_from_data_rise_until_data_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(DATA_VALID) |-> $stable(DATA_BYTES) until_with (DATA_VALID && DATA_READY);
endproperty

property stable_DATA_COUNT_from_data_rise_until_data_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(DATA_VALID) |-> $stable(DATA_COUNT) until_with (DATA_VALID && DATA_READY && DATA_LAST);
endproperty

property high_STATUS_VALID_from_status_rise_until_status_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(STATUS_VALID) |-> STATUS_VALID until_with (STATUS_VALID && STATUS_READY);
endproperty

property stable_STATUS_PID_from_status_rise_until_status_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(STATUS_VALID) |-> $stable(STATUS_PID) until_with (STATUS_VALID && STATUS_READY);
endproperty

property high_EP_BUSY_from_setup_hs_until_status_hs;
  @(posedge clk) disable iff (!rst_n)
    (SETUP_VALID && SETUP_READY) |-> EP_BUSY until_with (STATUS_VALID && STATUS_READY);
endproperty

property low_DATA_VALID_before_setup_hs;
  @(posedge clk) disable iff (!rst_n)
    !DATA_VALID until (SETUP_VALID && SETUP_READY);
endproperty

property low_STATUS_VALID_before_data_last;
  @(posedge clk) disable iff (!rst_n)
    !STATUS_VALID until (DATA_VALID && DATA_READY && DATA_LAST);
endproperty
