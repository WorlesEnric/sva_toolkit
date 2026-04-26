diagram td_burst_memory_td_000038 {
  clock posedge mclk;
  ticks 14;

  lane wdata: bus[16] = 7E 7E 7E 7E 7E 02 x x x x x x x x;
  lane rsp_valid: bit = 0 0 1 1 1 0 0 0 0 0 0 0 0 0;
  lane cmd_ready: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane cmd_valid: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane rsp_ready: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
  lane bit_0: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane bit_1: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
  lane rdata: bus[8] = x x 33 33 33 x x x x BE x x x x;
  lane addr: bus[32] = x x 00 00 00 x x x x x x F0 x x;
  lane bus_0: bus[8] = x 02 55 A1 01 55 x F0 x x x x x x;

  anchor first_beat = high(rsp_valid) and high(cmd_ready);
  anchor middle_beat = high(cmd_valid) and high(rsp_ready);
  anchor last_beat = high(bit_0) and high(bit_1) and eq(wdata, 02);

  window burst_first_to_middle = between first_beat and middle_beat [2:2];
  window burst_window = between middle_beat and last_beat [1:1];

  cut cut_between_38 = between burst_first_to_middle and burst_window compressed label "compressed lookback context";

  show high(rsp_valid) from first_beat until middle_beat;
  show stable(rdata) from first_beat until middle_beat;
  show stable(addr) from first_beat until middle_beat;
  show change(bus_0) at first_beat;
  show change(bus_0) at middle_beat;
  show change(bus_0) at last_beat;
}
