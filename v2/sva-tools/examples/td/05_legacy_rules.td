// Example 5: Multiple events with legacy rules
// Demonstrates: event keyword, legacy rule syntax (not-before, response, hold-until)

diagram legacy_rules {
  clock posedge clk;
  ticks 8;

  lane req:   bit = 0 0 1 1 1 1 0 0;
  lane ack:   bit = 0 0 0 0 0 1 0 0;
  lane valid: bit = 0 0 0 0 1 1 0 0;

  event req_start  = rise(req);
  event ack_seen   = rise(ack);
  event valid_seen = rise(valid);

  rule no_early_ack:
    not ack_seen before req_start;

  rule ack_after_req:
    req_start -> after [1:5] ack_seen;

  rule req_held:
    high(req) from req_start until ack_seen;
}
