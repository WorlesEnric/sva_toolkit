`define WIDTH 8
`ifdef SIM
`include "fixture_defs.svh"
`timescale 1ns/1ps
assert property (@(posedge clk) disable iff (!rst_n) req |-> ##1 ack);
`endif
