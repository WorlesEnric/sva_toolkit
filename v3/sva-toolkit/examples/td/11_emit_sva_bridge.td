// Example 11: Emit parameterized SVA from timing DSL
// Demonstrates: emit-sva bridge with a symbolic latency window

diagram emit_sva_bridge {
  clock posedge clk;
  disable iff !rst_n;

  param MAX_LATENCY;

  lane req: bit;
  lane ack: bit;

  anchor trigger = high(req);
  anchor response = high(ack);

  window response_window = between trigger and response [1:MAX_LATENCY];

  property req_to_ack:
    req |-> ##[1:MAX_LATENCY] ack;
}
