diagram beat_first_wait__bundle_23 {
  clock posedge clk;
  disable iff !rst_n;

  param BEAT_READY_MAX;
  param CMPL_MAX;
  param DESC_READY_MAX;
  param IRQ_ACK_MAX;
  param SRC_GNT_MAX;

  lane BEAT_READY: bit;
  lane BEAT_VALID: bit;
  lane CMPL_VALID: bit;
  lane LAST: bit;
  lane CMPL_READY: bit;
  lane SRC_GNT: bit;
  lane SRC_REQ: bit;
  lane DESC_READY: bit;
  lane DESC_VALID: bit;
  lane IRQ: bit;
  lane IRQ_ACK: bit;
  lane BYTE_EN: bus;
  lane CMPL_TAG: bus;
  lane DATA: bus;
  lane DESC_ADDR: bus;
  lane DESC_LEN: bus;
  lane DESC_TAG: bus;
  lane STATUS: bus;

  anchor beat_first_wait__bundle_23__node_0 = rise(BEAT_VALID);
  anchor beat_first_wait__bundle_23__point_1 = 1'b1;
  anchor beat_first_wait__bundle_23__node_2 = 1'b1;
  anchor beat_first_wait__bundle_23__node_3 = high(BEAT_VALID) and high(BEAT_READY);
  anchor beat_first_wait__bundle_23__node_4 = high(BEAT_VALID) and high(BEAT_READY) and high(LAST);
  anchor beat_first_wait__bundle_23__node_5 = rise(CMPL_VALID);
  anchor beat_first_wait__bundle_23__node_6 = rise(CMPL_VALID);
  anchor beat_first_wait__bundle_23__node_7 = high(CMPL_VALID) and high(CMPL_READY);
  anchor beat_first_wait__bundle_23__node_8 = high(BEAT_VALID) and high(BEAT_READY) and high(LAST);
  anchor beat_first_wait__bundle_23__node_9 = high(SRC_REQ) and high(SRC_GNT);
  anchor beat_first_wait__bundle_23__node_10 = rise(BEAT_VALID);
  anchor beat_first_wait__bundle_23__node_11 = rise(DESC_VALID);
  anchor beat_first_wait__bundle_23__node_12 = high(DESC_VALID) and high(DESC_READY);
  anchor beat_first_wait__bundle_23__node_13 = high(IRQ) and high(IRQ_ACK);
  anchor beat_first_wait__bundle_23__node_14 = high(IRQ_ACK);
  anchor beat_first_wait__bundle_23__node_15 = rise(IRQ);
  anchor beat_first_wait__bundle_23__node_16 = high(IRQ) and high(IRQ_ACK);
  anchor beat_first_wait__bundle_23__node_17 = rise(SRC_REQ);
  anchor beat_first_wait__bundle_23__node_18 = high(SRC_REQ) and high(SRC_GNT);
  anchor beat_first_wait__bundle_23__node_19 = high(CMPL_VALID) and high(CMPL_READY);
  anchor beat_first_wait__bundle_23__node_20 = rise(IRQ);
  anchor beat_first_wait__bundle_23__point_21 = 1'b1;

  window beat_first_wait__bundle_23__win_0 = between beat_first_wait__bundle_23__node_0 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_1 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_2 [0:0];
  window beat_first_wait__bundle_23__win_2 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_3 [0:BEAT_READY_MAX];
  window beat_first_wait__bundle_23__win_3 = between beat_first_wait__bundle_23__node_4 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_4 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_5 [1:CMPL_MAX];
  window beat_first_wait__bundle_23__win_5 = between beat_first_wait__bundle_23__node_6 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_6 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_7 [0:4];
  window beat_first_wait__bundle_23__win_7 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_8 [0:256];
  window beat_first_wait__bundle_23__win_8 = between beat_first_wait__bundle_23__node_9 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_9 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_10 [0:2];
  window beat_first_wait__bundle_23__win_10 = between beat_first_wait__bundle_23__node_11 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_11 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_12 [0:DESC_READY_MAX];
  window beat_first_wait__bundle_23__win_12 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_8 [0:0];
  window beat_first_wait__bundle_23__win_13 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_7 [0:0];
  window beat_first_wait__bundle_23__win_14 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_12 [0:0];
  window beat_first_wait__bundle_23__win_15 = between beat_first_wait__bundle_23__node_13 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_16 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_14 [0:0];
  window beat_first_wait__bundle_23__win_17 = between beat_first_wait__bundle_23__node_15 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_18 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_16 [0:0];
  window beat_first_wait__bundle_23__win_19 = between beat_first_wait__bundle_23__node_17 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_20 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_18 [0:0];
  window beat_first_wait__bundle_23__win_21 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_16 [0:IRQ_ACK_MAX];
  window beat_first_wait__bundle_23__win_22 = between beat_first_wait__bundle_23__node_19 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_23 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_20 [0:1];
  window beat_first_wait__bundle_23__win_24 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__point_1 [0:0];
  window beat_first_wait__bundle_23__win_25 = between beat_first_wait__bundle_23__node_2 and beat_first_wait__bundle_23__node_18 [0:SRC_GNT_MAX];
  window beat_first_wait__bundle_23__win_26 = between beat_first_wait__bundle_23__point_1 and beat_first_wait__bundle_23__node_3 [0:0];
  window beat_first_wait__bundle_23__win_27 = between beat_first_wait__bundle_23__node_3 and beat_first_wait__bundle_23__node_4 [0:0];
  window beat_first_wait__bundle_23__win_28 = between beat_first_wait__bundle_23__node_3 and beat_first_wait__bundle_23__node_0 [0:0];
  window beat_first_wait__bundle_23__win_29 = between beat_first_wait__bundle_23__node_5 and beat_first_wait__bundle_23__node_6 [0:0];
  window beat_first_wait__bundle_23__win_30 = between beat_first_wait__bundle_23__node_5 and beat_first_wait__bundle_23__node_19 [0:0];
  window beat_first_wait__bundle_23__win_31 = between beat_first_wait__bundle_23__node_7 and beat_first_wait__bundle_23__node_6 [0:0];
  window beat_first_wait__bundle_23__win_32 = between beat_first_wait__bundle_23__node_7 and beat_first_wait__bundle_23__node_19 [0:0];
  window beat_first_wait__bundle_23__win_33 = between beat_first_wait__bundle_23__node_8 and beat_first_wait__bundle_23__node_0 [0:0];
  window beat_first_wait__bundle_23__win_34 = between beat_first_wait__bundle_23__node_10 and beat_first_wait__bundle_23__node_0 [0:0];
  window beat_first_wait__bundle_23__win_35 = between beat_first_wait__bundle_23__node_12 and beat_first_wait__bundle_23__node_11 [0:0];
  window beat_first_wait__bundle_23__win_36 = between beat_first_wait__bundle_23__node_16 and beat_first_wait__bundle_23__node_15 [0:0];
  window beat_first_wait__bundle_23__win_37 = between beat_first_wait__bundle_23__node_18 and beat_first_wait__bundle_23__node_17 [0:0];
  window high_BEAT_VALID_from_beat_rise_until_beat_last__hold_2 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__node_4 [0:$];
  window high_CMPL_VALID_from_cpl_rise_until_cpl_handshake__hold_2 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__node_19 [0:$];
  window high_IRQ_from_irq_rise_until_irq_ack__hold_2 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__node_13 [0:$];
  window high_SRC_REQ_from_src_req_rise_until_src_grant__hold_2 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__node_9 [0:$];
  window low_BEAT_VALID_before_src_grant__hold_2 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__node_9 [0:$];
  window stable_CMPL_TAG_from_cpl_rise_until_cpl_handshake__hold_2 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__node_19 [0:$];
  window stable_STATUS_from_cpl_rise_until_cpl_handshake__hold_2 = between beat_first_wait__bundle_23__point_21 and beat_first_wait__bundle_23__node_19 [0:$];

  show high(BEAT_VALID) from beat_first_wait__bundle_23__point_21 until beat_first_wait__bundle_23__node_4;
  show high(CMPL_VALID) from beat_first_wait__bundle_23__point_21 until beat_first_wait__bundle_23__node_19;
  show high(IRQ) from beat_first_wait__bundle_23__point_21 until beat_first_wait__bundle_23__node_13;
  show high(SRC_REQ) from beat_first_wait__bundle_23__point_21 until beat_first_wait__bundle_23__node_9;
  show low(BEAT_VALID) from beat_first_wait__bundle_23__point_21 until beat_first_wait__bundle_23__node_9;
  show stable(CMPL_TAG) from beat_first_wait__bundle_23__point_21 until beat_first_wait__bundle_23__node_19;
  show stable(STATUS) from beat_first_wait__bundle_23__point_21 until beat_first_wait__bundle_23__node_19;

  property beat_first_wait:
    $rose(BEAT_VALID) |-> ##[0:BEAT_READY_MAX] BEAT_VALID && BEAT_READY;
  property cpl_latency:
    BEAT_VALID && BEAT_READY && LAST |-> ##[1:CMPL_MAX] $rose(CMPL_VALID);
  property cpl_wait:
    $rose(CMPL_VALID) |-> ##[0:4] CMPL_VALID && CMPL_READY;
  property data_phase:
    $rose(BEAT_VALID) |-> ##[0:256] BEAT_VALID && BEAT_READY && LAST;
  property data_start:
    SRC_REQ && SRC_GNT |-> ##[0:2] $rose(BEAT_VALID);
  property desc_wait:
    $rose(DESC_VALID) |-> ##[0:DESC_READY_MAX] DESC_VALID && DESC_READY;
  property high_BEAT_VALID_from_beat_rise_until_beat_last:
    $rose(BEAT_VALID) |-> BEAT_VALID until_with BEAT_VALID && BEAT_READY && LAST;
  property high_CMPL_VALID_from_cpl_rise_until_cpl_handshake:
    $rose(CMPL_VALID) |-> CMPL_VALID until_with CMPL_VALID && CMPL_READY;
  property high_DESC_VALID_from_desc_issue_until_desc_accept:
    $rose(DESC_VALID) |-> DESC_VALID until_with DESC_VALID && DESC_READY;
  property high_IRQ_ACK_at_irq_ack:
    IRQ && IRQ_ACK |-> IRQ_ACK;
  property high_IRQ_from_irq_rise_until_irq_ack:
    $rose(IRQ) |-> IRQ until_with IRQ && IRQ_ACK;
  property high_SRC_REQ_from_src_req_rise_until_src_grant:
    $rose(SRC_REQ) |-> SRC_REQ until_with SRC_REQ && SRC_GNT;
  property irq_hold:
    $rose(IRQ) |-> ##[0:IRQ_ACK_MAX] IRQ && IRQ_ACK;
  property irq_latency:
    CMPL_VALID && CMPL_READY |-> ##[0:1] $rose(IRQ);
  property low_BEAT_VALID_before_src_grant [lossy]:
    !BEAT_VALID until SRC_REQ && SRC_GNT;
  // low_BEAT_VALID_before_src_grant: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_BEAT_VALID_before_src_grant: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_BEAT_VALID_before_src_grant: until excludes the terminating cycle and is shown as a hold-until skeleton
  property src_acquire:
    $rose(SRC_REQ) |-> ##[0:SRC_GNT_MAX] SRC_REQ && SRC_GNT;
  property stable_BYTE_EN_from_beat_rise_until_beat_first:
    $rose(BEAT_VALID) |-> $stable(BYTE_EN) until_with BEAT_VALID && BEAT_READY;
  property stable_CMPL_TAG_from_cpl_rise_until_cpl_handshake:
    $rose(CMPL_VALID) |-> $stable(CMPL_TAG) until_with CMPL_VALID && CMPL_READY;
  property stable_DATA_from_beat_rise_until_beat_first:
    $rose(BEAT_VALID) |-> $stable(DATA) until_with BEAT_VALID && BEAT_READY;
  property stable_DESC_ADDR_from_desc_issue_until_desc_accept:
    $rose(DESC_VALID) |-> $stable(DESC_ADDR) until_with DESC_VALID && DESC_READY;
  property stable_DESC_LEN_from_desc_issue_until_desc_accept:
    $rose(DESC_VALID) |-> $stable(DESC_LEN) until_with DESC_VALID && DESC_READY;
  property stable_DESC_TAG_from_desc_issue_until_desc_accept:
    $rose(DESC_VALID) |-> $stable(DESC_TAG) until_with DESC_VALID && DESC_READY;
  property stable_STATUS_from_cpl_rise_until_cpl_handshake:
    $rose(CMPL_VALID) |-> $stable(STATUS) until_with CMPL_VALID && CMPL_READY;
}