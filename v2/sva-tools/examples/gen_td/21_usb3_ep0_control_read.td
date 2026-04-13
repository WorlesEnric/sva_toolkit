diagram data_first_gap__bundle_20 {
  clock posedge clk;
  disable iff !rst_n;

  param DATA_READY_MAX;
  param DATA_START_MAX;
  param SETUP_READY_MAX;
  param STATUS_READY_MAX;

  lane DATA_READY: bit;
  lane DATA_VALID: bit;
  lane DATA_LAST: bit;
  lane SETUP_READY: bit;
  lane SETUP_VALID: bit;
  lane EP_BUSY: bit;
  lane STATUS_READY: bit;
  lane STATUS_VALID: bit;
  lane BMREQ: bus;
  lane BREQ: bus;
  lane DATA_BYTES: bus;
  lane DATA_COUNT: bus;
  lane DATA_PID: bus;
  lane STATUS_PID: bus;
  lane WLEN: bus;
  lane WVALUE: bus;

  anchor data_first_gap__bundle_20__node_0 = rise(DATA_VALID);
  anchor data_first_gap__bundle_20__point_1 = 1'b1;
  anchor data_first_gap__bundle_20__node_2 = 1'b1;
  anchor data_first_gap__bundle_20__node_3 = high(DATA_VALID) and high(DATA_READY);
  anchor data_first_gap__bundle_20__node_4 = high(DATA_VALID) and high(DATA_READY) and high(DATA_LAST);
  anchor data_first_gap__bundle_20__node_5 = high(SETUP_VALID) and high(SETUP_READY);
  anchor data_first_gap__bundle_20__node_6 = rise(DATA_VALID);
  anchor data_first_gap__bundle_20__node_7 = high(STATUS_VALID) and high(STATUS_READY);
  anchor data_first_gap__bundle_20__node_8 = rise(SETUP_VALID);
  anchor data_first_gap__bundle_20__node_9 = high(SETUP_VALID) and high(SETUP_READY);
  anchor data_first_gap__bundle_20__node_10 = rise(STATUS_VALID);
  anchor data_first_gap__bundle_20__point_11 = 1'b1;
  anchor data_first_gap__bundle_20__node_12 = high(DATA_VALID) and high(DATA_READY) and high(DATA_LAST);
  anchor data_first_gap__bundle_20__node_13 = rise(STATUS_VALID);

  window data_first_gap__bundle_20__win_0 = between data_first_gap__bundle_20__node_0 and data_first_gap__bundle_20__point_1 [0:0];
  window data_first_gap__bundle_20__win_1 = between data_first_gap__bundle_20__point_1 and data_first_gap__bundle_20__node_2 [0:0];
  window data_first_gap__bundle_20__win_2 = between data_first_gap__bundle_20__node_2 and data_first_gap__bundle_20__node_3 [0:DATA_READY_MAX];
  window data_first_gap__bundle_20__win_3 = between data_first_gap__bundle_20__node_2 and data_first_gap__bundle_20__node_4 [0:8];
  window data_first_gap__bundle_20__win_4 = between data_first_gap__bundle_20__node_5 and data_first_gap__bundle_20__point_1 [0:0];
  window data_first_gap__bundle_20__win_5 = between data_first_gap__bundle_20__node_2 and data_first_gap__bundle_20__node_6 [1:DATA_START_MAX];
  window data_first_gap__bundle_20__win_6 = between data_first_gap__bundle_20__point_1 and data_first_gap__bundle_20__node_4 [0:0];
  window data_first_gap__bundle_20__win_7 = between data_first_gap__bundle_20__point_1 and data_first_gap__bundle_20__node_7 [0:0];
  window data_first_gap__bundle_20__win_8 = between data_first_gap__bundle_20__node_8 and data_first_gap__bundle_20__point_1 [0:0];
  window data_first_gap__bundle_20__win_9 = between data_first_gap__bundle_20__point_1 and data_first_gap__bundle_20__node_9 [0:0];
  window data_first_gap__bundle_20__win_10 = between data_first_gap__bundle_20__node_10 and data_first_gap__bundle_20__point_1 [0:0];
  window data_first_gap__bundle_20__win_11 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__point_1 [0:0];
  window data_first_gap__bundle_20__win_12 = between data_first_gap__bundle_20__node_2 and data_first_gap__bundle_20__node_9 [0:SETUP_READY_MAX];
  window data_first_gap__bundle_20__win_13 = between data_first_gap__bundle_20__point_1 and data_first_gap__bundle_20__node_3 [0:0];
  window data_first_gap__bundle_20__win_14 = between data_first_gap__bundle_20__node_12 and data_first_gap__bundle_20__point_1 [0:0];
  window data_first_gap__bundle_20__win_15 = between data_first_gap__bundle_20__node_2 and data_first_gap__bundle_20__node_13 [1:2];
  window data_first_gap__bundle_20__win_16 = between data_first_gap__bundle_20__node_2 and data_first_gap__bundle_20__node_7 [0:STATUS_READY_MAX];
  window data_first_gap__bundle_20__win_17 = between data_first_gap__bundle_20__node_3 and data_first_gap__bundle_20__node_0 [0:0];
  window data_first_gap__bundle_20__win_18 = between data_first_gap__bundle_20__node_3 and data_first_gap__bundle_20__node_12 [0:0];
  window data_first_gap__bundle_20__win_19 = between data_first_gap__bundle_20__node_4 and data_first_gap__bundle_20__node_0 [0:0];
  window data_first_gap__bundle_20__win_20 = between data_first_gap__bundle_20__node_4 and data_first_gap__bundle_20__node_12 [0:0];
  window data_first_gap__bundle_20__win_21 = between data_first_gap__bundle_20__node_6 and data_first_gap__bundle_20__node_0 [0:0];
  window data_first_gap__bundle_20__win_22 = between data_first_gap__bundle_20__node_6 and data_first_gap__bundle_20__node_12 [0:0];
  window data_first_gap__bundle_20__win_23 = between data_first_gap__bundle_20__node_7 and data_first_gap__bundle_20__node_10 [0:0];
  window data_first_gap__bundle_20__win_24 = between data_first_gap__bundle_20__node_9 and data_first_gap__bundle_20__node_8 [0:0];
  window data_first_gap__bundle_20__win_25 = between data_first_gap__bundle_20__node_13 and data_first_gap__bundle_20__node_10 [0:0];
  window high_DATA_VALID_from_data_rise_until_data_last__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_12 [0:$];
  window high_SETUP_VALID_from_setup_rise_until_setup_hs__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_5 [0:$];
  window low_DATA_VALID_before_setup_hs__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_5 [0:$];
  window low_STATUS_VALID_before_data_last__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_12 [0:$];
  window stable_BMREQ_from_setup_rise_until_setup_hs__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_5 [0:$];
  window stable_BREQ_from_setup_rise_until_setup_hs__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_5 [0:$];
  window stable_DATA_COUNT_from_data_rise_until_data_last__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_12 [0:$];
  window stable_DATA_PID_from_data_rise_until_data_last__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_12 [0:$];
  window stable_WLEN_from_setup_rise_until_setup_hs__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_5 [0:$];
  window stable_WVALUE_from_setup_rise_until_setup_hs__hold_2 = between data_first_gap__bundle_20__point_11 and data_first_gap__bundle_20__node_5 [0:$];

  show high(DATA_VALID) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_12;
  show high(SETUP_VALID) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_5;
  show low(DATA_VALID) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_5;
  show low(STATUS_VALID) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_12;
  show stable(BMREQ) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_5;
  show stable(BREQ) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_5;
  show stable(DATA_COUNT) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_12;
  show stable(DATA_PID) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_12;
  show stable(WLEN) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_5;
  show stable(WVALUE) from data_first_gap__bundle_20__point_11 until data_first_gap__bundle_20__node_5;

  property data_first_gap:
    $rose(DATA_VALID) |-> ##[0:DATA_READY_MAX] DATA_VALID && DATA_READY;
  property data_phase:
    $rose(DATA_VALID) |-> ##[0:8] DATA_VALID && DATA_READY && DATA_LAST;
  property data_start:
    SETUP_VALID && SETUP_READY |-> ##[1:DATA_START_MAX] $rose(DATA_VALID);
  property high_DATA_VALID_from_data_rise_until_data_last:
    $rose(DATA_VALID) |-> DATA_VALID until_with DATA_VALID && DATA_READY && DATA_LAST;
  property high_EP_BUSY_from_setup_hs_until_status_hs:
    SETUP_VALID && SETUP_READY |-> EP_BUSY until_with STATUS_VALID && STATUS_READY;
  property high_SETUP_VALID_from_setup_rise_until_setup_hs:
    $rose(SETUP_VALID) |-> SETUP_VALID until_with SETUP_VALID && SETUP_READY;
  property high_STATUS_VALID_from_status_rise_until_status_hs:
    $rose(STATUS_VALID) |-> STATUS_VALID until_with STATUS_VALID && STATUS_READY;
  property low_DATA_VALID_before_setup_hs [lossy]:
    !DATA_VALID until SETUP_VALID && SETUP_READY;
  // low_DATA_VALID_before_setup_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_DATA_VALID_before_setup_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_DATA_VALID_before_setup_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  property low_STATUS_VALID_before_data_last [lossy]:
    !STATUS_VALID until DATA_VALID && DATA_READY && DATA_LAST;
  // low_STATUS_VALID_before_data_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_STATUS_VALID_before_data_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_STATUS_VALID_before_data_last: until excludes the terminating cycle and is shown as a hold-until skeleton
  property setup_wait:
    $rose(SETUP_VALID) |-> ##[0:SETUP_READY_MAX] SETUP_VALID && SETUP_READY;
  property stable_BMREQ_from_setup_rise_until_setup_hs:
    $rose(SETUP_VALID) |-> $stable(BMREQ) until_with SETUP_VALID && SETUP_READY;
  property stable_BREQ_from_setup_rise_until_setup_hs:
    $rose(SETUP_VALID) |-> $stable(BREQ) until_with SETUP_VALID && SETUP_READY;
  property stable_DATA_BYTES_from_data_rise_until_data_first:
    $rose(DATA_VALID) |-> $stable(DATA_BYTES) until_with DATA_VALID && DATA_READY;
  property stable_DATA_COUNT_from_data_rise_until_data_last:
    $rose(DATA_VALID) |-> $stable(DATA_COUNT) until_with DATA_VALID && DATA_READY && DATA_LAST;
  property stable_DATA_PID_from_data_rise_until_data_last:
    $rose(DATA_VALID) |-> $stable(DATA_PID) until_with DATA_VALID && DATA_READY && DATA_LAST;
  property stable_STATUS_PID_from_status_rise_until_status_hs:
    $rose(STATUS_VALID) |-> $stable(STATUS_PID) until_with STATUS_VALID && STATUS_READY;
  property stable_WLEN_from_setup_rise_until_setup_hs:
    $rose(SETUP_VALID) |-> $stable(WLEN) until_with SETUP_VALID && SETUP_READY;
  property stable_WVALUE_from_setup_rise_until_setup_hs:
    $rose(SETUP_VALID) |-> $stable(WVALUE) until_with SETUP_VALID && SETUP_READY;
  property status_start:
    DATA_VALID && DATA_READY && DATA_LAST |-> ##[1:2] $rose(STATUS_VALID);
  property status_wait:
    $rose(STATUS_VALID) |-> ##[0:STATUS_READY_MAX] STATUS_VALID && STATUS_READY;
}