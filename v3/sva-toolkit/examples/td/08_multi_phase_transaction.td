// Example 8: Multi-phase transaction with chained windows
// Demonstrates: many anchors, chained windows, multiple properties, compound show

diagram multi_phase_transaction {
  clock posedge clk;
  ticks 12;

  lane req:    bit    = 0 1 1 1 1 1 1 1 1 0 0 0;
  lane addr:   bus[32] = x A0 A0 A0 A0 A0 A0 A0 A0 x x x;
  lane grant:  bit    = 0 0 0 1 1 1 1 0 0 0 0 0;
  lane data:   bus[64] = x x x x D0 D1 D2 x x x x x;
  lane valid:  bit    = 0 0 0 0 1 1 1 0 0 0 0 0;
  lane done:   bit    = 0 0 0 0 0 0 0 1 0 0 0 0;
  lane err:    bit    = 0 0 0 0 0 0 0 0 0 0 0 0;

  // Phase 1: arbitration
  anchor req_assert   = rise(req);
  anchor grant_assert = rise(grant);

  // Phase 2: data transfer
  anchor xfer_start = rise(valid);
  anchor xfer_end   = fall(valid);

  // Phase 3: completion
  anchor completion = rise(done);
  anchor error_flag = rise(err);

  window arb_phase  = between req_assert and grant_assert [1:4];
  window xfer_phase = between grant_assert and xfer_start [0:2];
  window data_phase = between xfer_start and xfer_end [1:8];
  window done_phase = between xfer_end and completion [0:1];

  cut before_req = before req_assert omitted;
  cut after_done = after completion omitted;

  show high(req) from req_assert until completion;
  show stable(addr) from req_assert until grant_assert;
  show high(grant) from grant_assert until xfer_end;
  show high(valid) from xfer_start until xfer_end;
  show low(err) from req_assert until completion;

  property req_hold:
    req |-> req until_with done;

  property addr_stable_during_arb:
    (req && !grant) |-> $stable(addr) until_with grant;

  property grant_before_data:
    req |-> ##[1:4] grant;

  property no_error:
    req |-> !err until_with done;
}
