// Example 6: Bus protocol with equality predicates
// Demonstrates: bus signals, eq/neq predicates, compound conditions, show in window

diagram bus_protocol {
  clock posedge clk;
  ticks 8;

  param RESP_OK;

  lane cmd:   bus[4] = x IDLE WR WR WR RD RD IDLE;
  lane resp:  bus[4] = x x x x OK x OK x;
  lane valid: bit    = 0 0 1 1 1 1 1 0;
  lane ready: bit    = 0 0 0 1 1 0 1 0;

  anchor write_start = eq(cmd, WR) and rise(valid);
  anchor write_done  = eq(cmd, WR) and high(ready);
  anchor read_start  = eq(cmd, RD) and high(valid);
  anchor resp_ok     = eq(resp, OK);

  window write_phase = between write_start and write_done [1:4];
  window read_phase  = between read_start and resp_ok [1:8];

  show stable(cmd) in write_phase;
  show high(valid) in write_phase;

  property write_completes:
    (valid && (cmd == WR)) |-> ##[1:4] (ready && (cmd == WR));

  property read_response:
    (valid && (cmd == RD)) |-> ##[1:8] (resp == RESP_OK);
}
