property core_wait(int CORE_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(CORE_RD_VALID) |-> ##[0:CORE_READY_MAX] (CORE_RD_VALID && CORE_RD_READY);
endproperty

property refresh_block(int REF_BLOCK_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(REF_PENDING) |-> ##[0:REF_BLOCK_MAX] (REF_PENDING && REF_ACK);
endproperty

property cmd_start;
  @(posedge clk) disable iff (!rst_n)
    (REF_PENDING && REF_ACK) |-> ##[1:2] $rose(CMD_VALID);
endproperty

property cmd_wait(int CMD_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(CMD_VALID) |-> ##[0:CMD_READY_MAX] (CMD_VALID && CMD_READY);
endproperty

property data_start(int DATA_START_MAX);
  @(posedge clk) disable iff (!rst_n)
    (CMD_VALID && CMD_READY) |-> ##[1:DATA_START_MAX] $rose(DQ_VALID);
endproperty

property data_first_gap(int DATA_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(DQ_VALID) |-> ##[0:DATA_READY_MAX] (DQ_VALID && DQ_READY);
endproperty

property data_phase;
  @(posedge clk) disable iff (!rst_n)
    $rose(DQ_VALID) |-> ##[0:8] (DQ_VALID && DQ_READY && DQ_LAST);
endproperty

property resp_start;
  @(posedge clk) disable iff (!rst_n)
    (DQ_VALID && DQ_READY && DQ_LAST) |-> ##[1:2] $rose(RESP_VALID);
endproperty

property resp_wait(int RESP_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(RESP_VALID) |-> ##[0:RESP_READY_MAX] (RESP_VALID && RESP_READY);
endproperty

property high_CORE_RD_VALID_from_core_req_rise_until_core_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CORE_RD_VALID) |-> CORE_RD_VALID until_with (CORE_RD_VALID && CORE_RD_READY);
endproperty

property stable_CORE_ADDR_from_core_req_rise_until_core_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CORE_RD_VALID) |-> $stable(CORE_ADDR) until_with (CORE_RD_VALID && CORE_RD_READY);
endproperty

property stable_CORE_LEN_from_core_req_rise_until_core_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CORE_RD_VALID) |-> $stable(CORE_LEN) until_with (CORE_RD_VALID && CORE_RD_READY);
endproperty

property stable_CORE_ID_from_core_req_rise_until_core_req_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CORE_RD_VALID) |-> $stable(CORE_ID) until_with (CORE_RD_VALID && CORE_RD_READY);
endproperty

property high_REF_PENDING_from_ref_rise_until_ref_done;
  @(posedge clk) disable iff (!rst_n)
    $rose(REF_PENDING) |-> REF_PENDING until_with (REF_PENDING && REF_ACK);
endproperty

property low_CMD_VALID_before_ref_done;
  @(posedge clk) disable iff (!rst_n)
    !CMD_VALID until (REF_PENDING && REF_ACK);
endproperty

property high_CMD_VALID_from_cmd_rise_until_cmd_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMD_VALID) |-> CMD_VALID until_with (CMD_VALID && CMD_READY);
endproperty

property stable_CMD_TYPE_from_cmd_rise_until_cmd_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMD_VALID) |-> $stable(CMD_TYPE) until_with (CMD_VALID && CMD_READY);
endproperty

property stable_CMD_ADDR_from_cmd_rise_until_cmd_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMD_VALID) |-> $stable(CMD_ADDR) until_with (CMD_VALID && CMD_READY);
endproperty

property stable_CMD_LEN_from_cmd_rise_until_cmd_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CMD_VALID) |-> $stable(CMD_LEN) until_with (CMD_VALID && CMD_READY);
endproperty

property high_DQ_VALID_from_dq_rise_until_dq_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(DQ_VALID) |-> DQ_VALID until_with (DQ_VALID && DQ_READY && DQ_LAST);
endproperty

property stable_DQ_DATA_from_dq_rise_until_dq_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(DQ_VALID) |-> $stable(DQ_DATA) until_with (DQ_VALID && DQ_READY);
endproperty

property stable_DQ_ID_from_dq_rise_until_dq_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(DQ_VALID) |-> $stable(DQ_ID) until_with (DQ_VALID && DQ_READY && DQ_LAST);
endproperty

property high_RESP_VALID_from_resp_rise_until_resp_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(RESP_VALID) |-> RESP_VALID until_with (RESP_VALID && RESP_READY);
endproperty

property stable_RESP_ID_from_resp_rise_until_resp_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(RESP_VALID) |-> $stable(RESP_ID) until_with (RESP_VALID && RESP_READY);
endproperty

property stable_RESP_STATUS_from_resp_rise_until_resp_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(RESP_VALID) |-> $stable(RESP_STATUS) until_with (RESP_VALID && RESP_READY);
endproperty

property low_RESP_VALID_before_dq_last;
  @(posedge clk) disable iff (!rst_n)
    !RESP_VALID until (DQ_VALID && DQ_READY && DQ_LAST);
endproperty
