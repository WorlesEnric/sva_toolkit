// Example 2: Signal stability constraint
// Demonstrates: bus lanes, show constraints, from-until region

diagram data_stable {
  clock posedge clk;
  ticks 6;

  lane valid: bit = 0 1 1 1 1 0;
  lane ready: bit = 0 0 0 0 1 0;
  lane data:  bus[8] = x A1 A1 A1 A1 x;

  anchor start = rise(valid);
  anchor done  = high(valid) and high(ready);

  window wait = between start and done [1:4];

  show high(valid) from start until done;
  show stable(data) from start until done;
}
