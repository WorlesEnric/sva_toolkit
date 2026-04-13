diagram a_wait__bundle_18 {
  clock posedge clk;
  disable iff !rst_n;

  param A_READY_MAX;
  param BURST_MAX;
  param D_READY_MAX;
  param D_START_MAX;

  lane A_READY: bit;
  lane A_VALID: bit;
  lane D_LAST: bit;
  lane D_READY: bit;
  lane D_VALID: bit;
  lane A_ADDRESS: bus;
  lane A_OPCODE: bus;
  lane A_SIZE: bus;
  lane A_SOURCE: bus;
  lane D_CORRUPT: bus;
  lane D_DATA: bus;
  lane D_DENIED: bus;
  lane D_OPCODE: bus;
  lane D_SOURCE: bus;

  anchor a_wait__bundle_18__node_0 = rise(A_VALID);
  anchor a_wait__bundle_18__point_1 = 1'b1;
  anchor a_wait__bundle_18__node_2 = 1'b1;
  anchor a_wait__bundle_18__node_3 = high(A_VALID) and high(A_READY);
  anchor a_wait__bundle_18__node_4 = rise(D_VALID);
  anchor a_wait__bundle_18__node_5 = high(D_VALID) and high(D_READY) and high(D_LAST);
  anchor a_wait__bundle_18__node_6 = high(D_VALID) and high(D_READY);
  anchor a_wait__bundle_18__node_7 = high(D_VALID) and high(D_READY);
  anchor a_wait__bundle_18__node_8 = high(A_VALID) and high(A_READY);
  anchor a_wait__bundle_18__node_9 = rise(D_VALID);
  anchor a_wait__bundle_18__node_10 = high(D_VALID) and high(D_READY) and high(D_LAST);
  anchor a_wait__bundle_18__node_11 = high(D_READY);
  anchor a_wait__bundle_18__point_12 = 1'b1;

  window a_wait__bundle_18__win_0 = between a_wait__bundle_18__node_0 and a_wait__bundle_18__point_1 [0:0];
  window a_wait__bundle_18__win_1 = between a_wait__bundle_18__point_1 and a_wait__bundle_18__node_2 [0:0];
  window a_wait__bundle_18__win_2 = between a_wait__bundle_18__node_2 and a_wait__bundle_18__node_3 [0:A_READY_MAX];
  window a_wait__bundle_18__win_3 = between a_wait__bundle_18__node_4 and a_wait__bundle_18__point_1 [0:0];
  window a_wait__bundle_18__win_4 = between a_wait__bundle_18__node_2 and a_wait__bundle_18__node_5 [0:BURST_MAX];
  window a_wait__bundle_18__win_5 = between a_wait__bundle_18__node_6 and a_wait__bundle_18__point_1 [0:0];
  window a_wait__bundle_18__win_6 = between a_wait__bundle_18__node_2 and a_wait__bundle_18__node_7 [0:D_READY_MAX];
  window a_wait__bundle_18__win_7 = between a_wait__bundle_18__node_8 and a_wait__bundle_18__point_1 [0:0];
  window a_wait__bundle_18__win_8 = between a_wait__bundle_18__node_2 and a_wait__bundle_18__node_9 [1:D_START_MAX];
  window a_wait__bundle_18__win_9 = between a_wait__bundle_18__point_1 and a_wait__bundle_18__node_3 [0:0];
  window a_wait__bundle_18__win_10 = between a_wait__bundle_18__node_10 and a_wait__bundle_18__point_1 [0:0];
  window a_wait__bundle_18__win_11 = between a_wait__bundle_18__point_1 and a_wait__bundle_18__node_11 [0:0];
  window a_wait__bundle_18__win_12 = between a_wait__bundle_18__point_1 and a_wait__bundle_18__node_5 [0:0];
  window a_wait__bundle_18__win_13 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__point_1 [0:0];
  window a_wait__bundle_18__win_14 = between a_wait__bundle_18__point_1 and a_wait__bundle_18__node_7 [0:0];
  window a_wait__bundle_18__win_15 = between a_wait__bundle_18__node_3 and a_wait__bundle_18__node_8 [0:0];
  window a_wait__bundle_18__win_16 = between a_wait__bundle_18__node_3 and a_wait__bundle_18__node_0 [0:0];
  window a_wait__bundle_18__win_17 = between a_wait__bundle_18__node_5 and a_wait__bundle_18__node_6 [0:0];
  window a_wait__bundle_18__win_18 = between a_wait__bundle_18__node_5 and a_wait__bundle_18__node_4 [0:0];
  window a_wait__bundle_18__win_19 = between a_wait__bundle_18__node_5 and a_wait__bundle_18__node_10 [0:0];
  window a_wait__bundle_18__win_20 = between a_wait__bundle_18__node_7 and a_wait__bundle_18__node_10 [0:0];
  window a_wait__bundle_18__win_21 = between a_wait__bundle_18__node_7 and a_wait__bundle_18__node_4 [0:0];
  window a_wait__bundle_18__win_22 = between a_wait__bundle_18__node_9 and a_wait__bundle_18__node_10 [0:0];
  window a_wait__bundle_18__win_23 = between a_wait__bundle_18__node_9 and a_wait__bundle_18__node_4 [0:0];
  window high_A_VALID_from_a_get_rise_until_a_handshake__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_8 [0:$];
  window high_D_VALID_from_d_valid_rise_until_d_last_beat__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_10 [0:$];
  window low_D_VALID_before_a_handshake__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_8 [0:$];
  window stable_A_ADDRESS_from_a_get_rise_until_a_handshake__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_8 [0:$];
  window stable_A_OPCODE_from_a_get_rise_until_a_handshake__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_8 [0:$];
  window stable_A_SIZE_from_a_get_rise_until_a_handshake__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_8 [0:$];
  window stable_A_SOURCE_from_a_get_rise_until_a_handshake__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_8 [0:$];
  window stable_D_CORRUPT_from_d_valid_rise_until_d_last_beat__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_10 [0:$];
  window stable_D_DATA_from_d_valid_rise_until_d_first_beat__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_6 [0:$];
  window stable_D_DENIED_from_d_valid_rise_until_d_last_beat__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_10 [0:$];
  window stable_D_OPCODE_from_d_valid_rise_until_d_last_beat__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_10 [0:$];
  window stable_D_SOURCE_from_d_valid_rise_until_d_last_beat__hold_2 = between a_wait__bundle_18__point_12 and a_wait__bundle_18__node_10 [0:$];

  show high(A_VALID) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_8;
  show high(D_VALID) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_10;
  show low(D_VALID) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_8;
  show stable(A_ADDRESS) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_8;
  show stable(A_OPCODE) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_8;
  show stable(A_SIZE) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_8;
  show stable(A_SOURCE) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_8;
  show stable(D_CORRUPT) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_10;
  show stable(D_DATA) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_6;
  show stable(D_DENIED) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_10;
  show stable(D_OPCODE) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_10;
  show stable(D_SOURCE) from a_wait__bundle_18__point_12 until a_wait__bundle_18__node_10;

  property a_wait:
    $rose(A_VALID) |-> ##[0:A_READY_MAX] A_VALID && A_READY;
  property d_burst_phase:
    $rose(D_VALID) |-> ##[0:BURST_MAX] D_VALID && D_READY && D_LAST;
  property d_completion:
    D_VALID && D_READY |-> ##[0:BURST_MAX] D_VALID && D_READY && D_LAST;
  property d_first_wait:
    $rose(D_VALID) |-> ##[0:D_READY_MAX] D_VALID && D_READY;
  property d_start:
    A_VALID && A_READY |-> ##[1:D_START_MAX] $rose(D_VALID);
  property high_A_VALID_from_a_get_rise_until_a_handshake:
    $rose(A_VALID) |-> A_VALID until_with A_VALID && A_READY;
  property high_D_READY_at_d_last_beat:
    D_VALID && D_READY && D_LAST |-> D_READY;
  property high_D_VALID_from_d_valid_rise_until_d_last_beat:
    $rose(D_VALID) |-> D_VALID until_with D_VALID && D_READY && D_LAST;
  property low_D_VALID_before_a_handshake [lossy]:
    !D_VALID until A_VALID && A_READY;
  // low_D_VALID_before_a_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_D_VALID_before_a_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  // low_D_VALID_before_a_handshake: until excludes the terminating cycle and is shown as a hold-until skeleton
  property stable_A_ADDRESS_from_a_get_rise_until_a_handshake:
    $rose(A_VALID) |-> $stable(A_ADDRESS) until_with A_VALID && A_READY;
  property stable_A_OPCODE_from_a_get_rise_until_a_handshake:
    $rose(A_VALID) |-> $stable(A_OPCODE) until_with A_VALID && A_READY;
  property stable_A_SIZE_from_a_get_rise_until_a_handshake:
    $rose(A_VALID) |-> $stable(A_SIZE) until_with A_VALID && A_READY;
  property stable_A_SOURCE_from_a_get_rise_until_a_handshake:
    $rose(A_VALID) |-> $stable(A_SOURCE) until_with A_VALID && A_READY;
  property stable_D_CORRUPT_from_d_valid_rise_until_d_last_beat:
    $rose(D_VALID) |-> $stable(D_CORRUPT) until_with D_VALID && D_READY && D_LAST;
  property stable_D_DATA_from_d_valid_rise_until_d_first_beat:
    $rose(D_VALID) |-> $stable(D_DATA) until_with D_VALID && D_READY;
  property stable_D_DENIED_from_d_valid_rise_until_d_last_beat:
    $rose(D_VALID) |-> $stable(D_DENIED) until_with D_VALID && D_READY && D_LAST;
  property stable_D_OPCODE_from_d_valid_rise_until_d_last_beat:
    $rose(D_VALID) |-> $stable(D_OPCODE) until_with D_VALID && D_READY && D_LAST;
  property stable_D_SOURCE_from_d_valid_rise_until_d_last_beat:
    $rose(D_VALID) |-> $stable(D_SOURCE) until_with D_VALID && D_READY && D_LAST;
}