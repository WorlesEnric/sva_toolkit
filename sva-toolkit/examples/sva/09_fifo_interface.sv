property fill_phase(int DEPTH);
  @(posedge clk) disable iff (!rst_n)
    ($rose(push) && !full) |-> ##[1:DEPTH] $rose(full);
endproperty

property drain_phase(int DEPTH);
  @(posedge clk) disable iff (!rst_n)
    ($rose(pop) && !empty) |-> ##[1:DEPTH] $rose(empty);
endproperty

property data_latency;
  @(posedge clk) disable iff (!rst_n)
    ($rose(push) && !full) |-> ##[1:$] ($rose(pop) && !empty);
endproperty

property low_full_at_first_push;
  @(posedge clk) disable iff (!rst_n)
    ($rose(push) && !full) |-> !full;
endproperty

property low_empty_at_first_pop;
  @(posedge clk) disable iff (!rst_n)
    ($rose(pop) && !empty) |-> !empty;
endproperty

property low_pop_from_first_push_until_first_pop;
  @(posedge clk) disable iff (!rst_n)
    ($rose(push) && !full) |-> !pop until_with ($rose(pop) && !empty);
endproperty

property low_push_from_first_pop_until_fifo_empty;
  @(posedge clk) disable iff (!rst_n)
    ($rose(pop) && !empty) |-> !push until_with $rose(empty);
endproperty
