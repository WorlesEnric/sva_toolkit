diagram td_parallel_noc_td_000050 {
  clock posedge clk;
  ticks 17;

  lane CR_READY: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane CR_VALID: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RX_READY: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_READY: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RX_VALID: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_HDR: bus[16] = x BE BE x x x x x x x A1 x x x x x x;
  lane RX_DATA: bus[32] = x 02 FF x x x x x x x x x x x x x x;
  lane TX_VALID: bit = 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;

  anchor a_start = rise(CR_READY);
  anchor a_end = high(CR_VALID);
  anchor b_start = rise(RX_READY);
  anchor b_end = high(TX_READY) and high(RX_VALID);

  window a_window = between a_start and a_end [1:1];
  window b_window = between b_start and b_end [1:1];

  show stable(TX_HDR) from a_start until a_end;
  show low(CR_VALID) before a_start;
  show neq(RX_DATA, BE) at b_end;
  show high(TX_VALID) after b_end;
}
