/* outer /* nested */
assert property (
  /* clock */ @(posedge clk)
  /* reset */ disable iff (!rst_n)
  req /* delay */ |-> ##1 ack
);
