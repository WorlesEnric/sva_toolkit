diagram td_fork_generic_td_000000 {
  clock negedge clk;
  ticks 16;

  param MAX_LAT;
  param READY_MAX;

  lane addr: bus[16] = 01 01 EE x x x x x x x x x x x x x;
  lane start: bit = 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
  lane resp_valid: bit = 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
  lane id: bus[16] = x x 1F 1F 1F x x x x x x x x x x A5;
  lane stop: bit = 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0;
  lane len: bus[16] = x x EE EE EE x x x x 55 x x x x x x;
  lane tag: bus[16] = 00 A5 x x x 01 x x x x x x x x x x;
  lane valid: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane data: bus[8] = x x 00 00 00 x x x x x x EE x x x x;
  lane busy: bit = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane req: bit = 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;

  anchor trigger = change(addr);
  anchor branch_a = fall(start);
  anchor branch_b = rise(resp_valid);

  window fork_a = between trigger and branch_a [1:READY_MAX];
  window fork_b = between trigger and branch_b [1:MAX_LAT];

  cut cut_after_0 = after branch_a omitted;

  show stable(id) from trigger until branch_a;
  show high(stop) from trigger until branch_b;
  show stable(len) from trigger until branch_a;
  show neq(tag, 33) before trigger;
  show fall(valid) at branch_b;
  show stable(data) in fork_a;
  show low(busy) after branch_b;
}
