// leading comment
assert property ( // statement comment
  @(posedge clk) // clock comment
  disable iff (!rst_n)
  req // antecedent comment
  |-> ##1 ack
); // trailing comment
