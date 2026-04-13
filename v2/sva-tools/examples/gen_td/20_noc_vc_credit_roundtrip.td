diagram cr_latency__bundle_19 {
  clock posedge clk;
  disable iff !rst_n;

  param CREDIT_LAT_MAX;
  param RX_READY_MAX;
  param RX_START_MAX;
  param TX_READY_MAX;

  lane CR_VALID: bit;
  lane RX_READY: bit;
  lane RX_TAIL: bit;
  lane RX_VALID: bit;
  lane CR_READY: bit;
  lane TX_READY: bit;
  lane TX_VALID: bit;
  lane CR_COUNT: bus;
  lane CR_VC: bus;
  lane RX_CLASS: bus;
  lane RX_DATA: bus;
  lane RX_VC: bus;
  lane TX_CLASS: bus;
  lane TX_DEST: bus;
  lane TX_HDR: bus;
  lane TX_VC: bus;

  anchor cr_latency__bundle_19__node_0 = high(RX_VALID) and high(RX_READY) and high(RX_TAIL);
  anchor cr_latency__bundle_19__point_1 = 1'b1;
  anchor cr_latency__bundle_19__node_2 = 1'b1;
  anchor cr_latency__bundle_19__node_3 = rise(CR_VALID);
  anchor cr_latency__bundle_19__node_4 = rise(CR_VALID);
  anchor cr_latency__bundle_19__node_5 = high(CR_VALID) and high(CR_READY);
  anchor cr_latency__bundle_19__node_6 = rise(RX_VALID);
  anchor cr_latency__bundle_19__node_7 = high(RX_VALID) and high(RX_READY) and high(RX_TAIL);
  anchor cr_latency__bundle_19__node_8 = rise(TX_VALID);
  anchor cr_latency__bundle_19__node_9 = high(TX_VALID) and high(TX_READY);
  anchor cr_latency__bundle_19__point_10 = 1'b1;
  anchor cr_latency__bundle_19__node_11 = high(RX_VALID) and high(RX_READY);
  anchor cr_latency__bundle_19__node_12 = high(TX_VALID) and high(TX_READY);
  anchor cr_latency__bundle_19__node_13 = rise(RX_VALID);

  window cr_latency__bundle_19__win_0 = between cr_latency__bundle_19__node_0 and cr_latency__bundle_19__point_1 [0:0];
  window cr_latency__bundle_19__win_1 = between cr_latency__bundle_19__point_1 and cr_latency__bundle_19__node_2 [0:0];
  window cr_latency__bundle_19__win_2 = between cr_latency__bundle_19__node_2 and cr_latency__bundle_19__node_3 [0:CREDIT_LAT_MAX];
  window cr_latency__bundle_19__win_3 = between cr_latency__bundle_19__node_4 and cr_latency__bundle_19__point_1 [0:0];
  window cr_latency__bundle_19__win_4 = between cr_latency__bundle_19__node_2 and cr_latency__bundle_19__node_5 [0:1];
  window cr_latency__bundle_19__win_5 = between cr_latency__bundle_19__point_1 and cr_latency__bundle_19__node_5 [0:0];
  window cr_latency__bundle_19__win_6 = between cr_latency__bundle_19__node_6 and cr_latency__bundle_19__point_1 [0:0];
  window cr_latency__bundle_19__win_7 = between cr_latency__bundle_19__point_1 and cr_latency__bundle_19__node_7 [0:0];
  window cr_latency__bundle_19__win_8 = between cr_latency__bundle_19__node_8 and cr_latency__bundle_19__point_1 [0:0];
  window cr_latency__bundle_19__win_9 = between cr_latency__bundle_19__point_1 and cr_latency__bundle_19__node_9 [0:0];
  window cr_latency__bundle_19__win_10 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__point_1 [0:0];
  window cr_latency__bundle_19__win_11 = between cr_latency__bundle_19__node_2 and cr_latency__bundle_19__node_11 [0:RX_READY_MAX];
  window cr_latency__bundle_19__win_12 = between cr_latency__bundle_19__node_2 and cr_latency__bundle_19__node_7 [0:8];
  window cr_latency__bundle_19__win_13 = between cr_latency__bundle_19__node_12 and cr_latency__bundle_19__point_1 [0:0];
  window cr_latency__bundle_19__win_14 = between cr_latency__bundle_19__node_2 and cr_latency__bundle_19__node_13 [1:RX_START_MAX];
  window cr_latency__bundle_19__win_15 = between cr_latency__bundle_19__point_1 and cr_latency__bundle_19__node_11 [0:0];
  window cr_latency__bundle_19__win_16 = between cr_latency__bundle_19__node_2 and cr_latency__bundle_19__node_9 [0:TX_READY_MAX];
  window cr_latency__bundle_19__win_17 = between cr_latency__bundle_19__node_3 and cr_latency__bundle_19__node_4 [0:0];
  window cr_latency__bundle_19__win_18 = between cr_latency__bundle_19__node_5 and cr_latency__bundle_19__node_4 [0:0];
  window cr_latency__bundle_19__win_19 = between cr_latency__bundle_19__node_7 and cr_latency__bundle_19__node_6 [0:0];
  window cr_latency__bundle_19__win_20 = between cr_latency__bundle_19__node_9 and cr_latency__bundle_19__node_12 [0:0];
  window cr_latency__bundle_19__win_21 = between cr_latency__bundle_19__node_9 and cr_latency__bundle_19__node_8 [0:0];
  window cr_latency__bundle_19__win_22 = between cr_latency__bundle_19__node_11 and cr_latency__bundle_19__node_6 [0:0];
  window cr_latency__bundle_19__win_23 = between cr_latency__bundle_19__node_13 and cr_latency__bundle_19__node_6 [0:0];
  window high_RX_VALID_from_rx_rise_until_rx_last__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_0 [0:$];
  window high_TX_VALID_from_tx_rise_until_tx_hs__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_12 [0:$];
  window low_RX_VALID_before_tx_hs__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_12 [0:$];
  window stable_RX_CLASS_from_rx_rise_until_rx_last__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_0 [0:$];
  window stable_RX_VC_from_rx_rise_until_rx_last__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_0 [0:$];
  window stable_TX_CLASS_from_tx_rise_until_tx_hs__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_12 [0:$];
  window stable_TX_DEST_from_tx_rise_until_tx_hs__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_12 [0:$];
  window stable_TX_HDR_from_tx_rise_until_tx_hs__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_12 [0:$];
  window stable_TX_VC_from_tx_rise_until_tx_hs__hold_2 = between cr_latency__bundle_19__point_10 and cr_latency__bundle_19__node_12 [0:$];

  show high(RX_VALID) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_0;
  show high(TX_VALID) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_12;
  show low(RX_VALID) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_12;
  show stable(RX_CLASS) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_0;
  show stable(RX_VC) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_0;
  show stable(TX_CLASS) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_12;
  show stable(TX_DEST) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_12;
  show stable(TX_HDR) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_12;
  show stable(TX_VC) from cr_latency__bundle_19__point_10 until cr_latency__bundle_19__node_12;

  property cr_latency:
    RX_VALID && RX_READY && RX_TAIL |-> ##[0:CREDIT_LAT_MAX] $rose(CR_VALID);
  property cr_wait:
    $rose(CR_VALID) |-> ##[0:1] CR_VALID && CR_READY;
  property high_CR_VALID_from_cr_rise_until_cr_hs:
    $rose(CR_VALID) |-> CR_VALID until_with CR_VALID && CR_READY;
  property high_RX_VALID_from_rx_rise_until_rx_last:
    $rose(RX_VALID) |-> RX_VALID until_with RX_VALID && RX_READY && RX_TAIL;
  property high_TX_VALID_from_tx_rise_until_tx_hs:
    $rose(TX_VALID) |-> TX_VALID until_with TX_VALID && TX_READY;
  property low_RX_VALID_before_tx_hs [lossy]:
    !RX_VALID until TX_VALID && TX_READY;
  // low_RX_VALID_before_tx_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RX_VALID_before_tx_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_RX_VALID_before_tx_hs: until excludes the terminating cycle and is shown as a hold-until skeleton
  property rx_first_wait:
    $rose(RX_VALID) |-> ##[0:RX_READY_MAX] RX_VALID && RX_READY;
  property rx_phase:
    $rose(RX_VALID) |-> ##[0:8] RX_VALID && RX_READY && RX_TAIL;
  property rx_start:
    TX_VALID && TX_READY |-> ##[1:RX_START_MAX] $rose(RX_VALID);
  property stable_CR_COUNT_from_cr_rise_until_cr_hs:
    $rose(CR_VALID) |-> $stable(CR_COUNT) until_with CR_VALID && CR_READY;
  property stable_CR_VC_from_cr_rise_until_cr_hs:
    $rose(CR_VALID) |-> $stable(CR_VC) until_with CR_VALID && CR_READY;
  property stable_RX_CLASS_from_rx_rise_until_rx_last:
    $rose(RX_VALID) |-> $stable(RX_CLASS) until_with RX_VALID && RX_READY && RX_TAIL;
  property stable_RX_DATA_from_rx_rise_until_rx_first:
    $rose(RX_VALID) |-> $stable(RX_DATA) until_with RX_VALID && RX_READY;
  property stable_RX_VC_from_rx_rise_until_rx_last:
    $rose(RX_VALID) |-> $stable(RX_VC) until_with RX_VALID && RX_READY && RX_TAIL;
  property stable_TX_CLASS_from_tx_rise_until_tx_hs:
    $rose(TX_VALID) |-> $stable(TX_CLASS) until_with TX_VALID && TX_READY;
  property stable_TX_DEST_from_tx_rise_until_tx_hs:
    $rose(TX_VALID) |-> $stable(TX_DEST) until_with TX_VALID && TX_READY;
  property stable_TX_HDR_from_tx_rise_until_tx_hs:
    $rose(TX_VALID) |-> $stable(TX_HDR) until_with TX_VALID && TX_READY;
  property stable_TX_VC_from_tx_rise_until_tx_hs:
    $rose(TX_VALID) |-> $stable(TX_VC) until_with TX_VALID && TX_READY;
  property tx_wait:
    $rose(TX_VALID) |-> ##[0:TX_READY_MAX] TX_VALID && TX_READY;
}