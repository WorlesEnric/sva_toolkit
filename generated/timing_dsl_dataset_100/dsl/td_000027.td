diagram td_parallel_noc_td_000027 {
  clock negedge clk;
  ticks 20;

  lane CR_VALID: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane RX_VALID: bit = 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_READY: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_VALID: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane VC: bus[32] = x 0A 0A 0A 0A 0A x x x x x x F0 x x x x x x x;
  lane RX_DATA: bus[16] = x 7E 7E 7E 7E 7E x x x x x 55 x x x x x x x x;
  lane CR_READY: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane TX_HDR: bus[8] = x x x x 02 x x x x x x x x x x x x FF x x;
  lane TX_DATA: bus[32] = 1F FF x x x x x x x x x x x x x x 02 x x x;

  anchor a_start = rise(CR_VALID);
  anchor a_end = low(RX_VALID);
  anchor b_start = rise(TX_READY);
  anchor b_end = rise(TX_VALID);

  window a_window = between a_start and a_end [2:4];
  window b_window = between b_start and b_end [2:4];

  show VC = 0A from a_start until a_end;
  show stable(RX_DATA) in a_window;
  show low(CR_READY) before b_end;
  show TX_HDR = 02 at b_end;
  show change(TX_DATA) in a_window;
}
