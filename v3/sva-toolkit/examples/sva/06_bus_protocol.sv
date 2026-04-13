property write_phase;
  @(posedge clk)
    (cmd == WR && $rose(valid)) |-> ##[1:4] (cmd == WR && ready);
endproperty

property read_phase;
  @(posedge clk)
    (cmd == RD && valid) |-> ##[1:8] resp == OK;
endproperty
