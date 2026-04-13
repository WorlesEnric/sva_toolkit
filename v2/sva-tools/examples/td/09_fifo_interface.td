// Example 9: FIFO interface with multiple constraint types
// Demonstrates: complex conditions (and/or/not), at/in/before/after regions,
//               unbounded window, multiple bus lanes, unsupported property

diagram fifo_interface {
  clock posedge clk;
  disable iff !rst_n;
  ticks 10;

  param DEPTH;

  lane push:         bit    = 0 1 1 1 0 0 0 0 0 0;
  lane pop:          bit    = 0 0 0 0 0 1 1 1 0 0;
  lane wdata:        bus[8] = x AA BB CC x x x x x x;
  lane rdata:        bus[8] = x x x x x AA BB CC x x;
  lane full:         bit    = 0 0 0 0 1 0 0 0 0 0;
  lane empty:        bit    = 1 0 0 0 0 0 0 0 1 1;
  lane almost_full:  bit    = 0 0 0 1 1 0 0 0 0 0;
  lane almost_empty: bit    = 1 1 0 0 0 0 0 1 1 1;

  anchor first_push  = rise(push) and low(full);
  anchor fifo_full   = rise(full);
  anchor first_pop   = rise(pop) and low(empty);
  anchor fifo_empty  = rise(empty);
  anchor overflow    = high(push) and high(full);
  anchor underflow   = high(pop) and high(empty);

  window fill_phase   = between first_push and fifo_full [1:DEPTH];
  window drain_phase  = between first_pop and fifo_empty [1:DEPTH];
  window data_latency = between first_push and first_pop [1:$];

  cut before_ops = before first_push omitted label "init";
  cut after_drain = after fifo_empty omitted;

  show low(full) at first_push;
  show low(empty) at first_pop;
  show high(push) in fill_phase;
  show high(pop) in drain_phase;
  show low(pop) from first_push until first_pop;
  show low(push) from first_pop until fifo_empty;

  property no_overflow:
    full |-> !push;

  property no_underflow:
    empty |-> !pop;

  property fifo_ordering:
    (push && !full) |-> ##[1:$] (pop && (rdata == wdata));

  property almost_full_warning [lossy]:
    ($rose(almost_full)) |-> ##[1:DEPTH] full || $fell(push);
  // almost_full_warning: depends on push/pop interleaving

  property fifo_depth_invariant [lossy]:
    (push && !pop && !full) |-> ##1 (count == ($past(count) + 1));
  // fifo_depth_invariant: internal count signal not directly observable
}
