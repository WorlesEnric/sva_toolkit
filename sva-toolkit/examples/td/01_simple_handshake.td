// Example 1: Simple two-signal handshake
// Demonstrates: clock, ticks, bit lanes with samples, one anchor, one window

// `diagram` keyword declares a new timing scenario named `simple_handshake`.
diagram simple_handshake {
  // `clock` keyword specifies the sampling event. `posedge clk` means signals are sampled on the rising edge of `clk`.
  clock posedge clk;
  
  // `ticks` keyword sets the horizontal length of the diagram to 6 clock cycles.
  ticks 6;

  // `lane` defines a signal track. `req` is the name, `bit` is the type, and `0 0 1...` are the sample values per tick.
  lane req: bit = 0 0 1 1 1 0;
  
  // Similarly, `ack` lane shows the acknowledgment signal's states over the 6 ticks.
  lane ack: bit = 0 0 0 0 1 0;

  // `anchor` defines a named point in time. `req_rise` points to the tick where `req` transitions from 0 to 1.
  anchor req_rise = rise(req);
  
  // `ack_rise` points to the tick where `ack` transitions from 0 to 1.
  anchor ack_rise = rise(ack);

  // `window` creates a visual range between two anchors. Here, `response_time` spans from `req_rise` to `ack_rise` and implies a constraint of 1 to 4 cycles.
  window response_time = between req_rise and ack_rise [1:4];
}
