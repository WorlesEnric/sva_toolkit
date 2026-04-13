property a_wait(int A_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(A_VALID) |-> ##[0:A_READY_MAX] (A_VALID && A_READY);
endproperty

property d_start(int D_START_MAX);
  @(posedge clk) disable iff (!rst_n)
    (A_VALID && A_READY) |-> ##[1:D_START_MAX] $rose(D_VALID);
endproperty

property d_first_wait(int D_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> ##[0:D_READY_MAX] (D_VALID && D_READY);
endproperty

property d_burst_phase(int BURST_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> ##[0:BURST_MAX] (D_VALID && D_READY && D_LAST);
endproperty

property d_completion(int BURST_MAX);
  @(posedge clk) disable iff (!rst_n)
    (D_VALID && D_READY) |-> ##[0:BURST_MAX] (D_VALID && D_READY && D_LAST);
endproperty

property high_A_VALID_from_a_get_rise_until_a_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(A_VALID) |-> A_VALID until_with (A_VALID && A_READY);
endproperty

property stable_A_OPCODE_from_a_get_rise_until_a_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(A_VALID) |-> $stable(A_OPCODE) until_with (A_VALID && A_READY);
endproperty

property stable_A_ADDRESS_from_a_get_rise_until_a_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(A_VALID) |-> $stable(A_ADDRESS) until_with (A_VALID && A_READY);
endproperty

property stable_A_SIZE_from_a_get_rise_until_a_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(A_VALID) |-> $stable(A_SIZE) until_with (A_VALID && A_READY);
endproperty

property stable_A_SOURCE_from_a_get_rise_until_a_handshake;
  @(posedge clk) disable iff (!rst_n)
    $rose(A_VALID) |-> $stable(A_SOURCE) until_with (A_VALID && A_READY);
endproperty

property high_D_VALID_from_d_valid_rise_until_d_last_beat;
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> D_VALID until_with (D_VALID && D_READY && D_LAST);
endproperty

property stable_D_DATA_from_d_valid_rise_until_d_first_beat;
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> $stable(D_DATA) until_with (D_VALID && D_READY);
endproperty

property stable_D_OPCODE_from_d_valid_rise_until_d_last_beat;
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> $stable(D_OPCODE) until_with (D_VALID && D_READY && D_LAST);
endproperty

property stable_D_SOURCE_from_d_valid_rise_until_d_last_beat;
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> $stable(D_SOURCE) until_with (D_VALID && D_READY && D_LAST);
endproperty

property stable_D_DENIED_from_d_valid_rise_until_d_last_beat;
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> $stable(D_DENIED) until_with (D_VALID && D_READY && D_LAST);
endproperty

property stable_D_CORRUPT_from_d_valid_rise_until_d_last_beat;
  @(posedge clk) disable iff (!rst_n)
    $rose(D_VALID) |-> $stable(D_CORRUPT) until_with (D_VALID && D_READY && D_LAST);
endproperty

property low_D_VALID_before_a_handshake;
  @(posedge clk) disable iff (!rst_n)
    !D_VALID until (A_VALID && A_READY);
endproperty

property high_D_READY_at_d_last_beat;
  @(posedge clk) disable iff (!rst_n)
    (D_VALID && D_READY && D_LAST) |-> D_READY;
endproperty
