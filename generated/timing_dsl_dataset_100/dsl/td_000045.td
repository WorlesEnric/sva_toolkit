diagram td_single_response_generic_td_000045 {
  clock negedge clk;
  ticks 14;

  lane start: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane grant: bit = 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane data: bus[32] = x x 10 10 x x x x x x x 1F x x;
  lane tag: bus[8] = A5 01 x x x FF x x x x x x x x;
  lane ack: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = high(start);
  anchor response = rise(grant);

  window response_window = between trigger and response [1:1];

  show stable(data) from trigger until response;
  show neq(tag, F0) before trigger;
  show fall(ack) at response;
}
