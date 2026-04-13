diagram read_phase__write_phase {
  clock posedge clk;
  ticks 10;

  param RD;
  param OK;
  param WR;

  lane cmd: bus = 0 0 0 0 0 = 0 = 0 0;
  lane resp: bus = x 0 0 0 0 = x 0 x 0;
  lane valid: bit = 1 1 1 0 1 1 1 1 1 1;
  lane ready: bit = x 1 1 1 1 1 x 1 x 1;

  anchor read_phase__write_phase__node_0 = eq(cmd, RD) and high(valid);
  anchor read_phase__write_phase__point_1 = 1'b1;
  anchor read_phase__write_phase__node_2 = 1'b1;
  anchor read_phase__write_phase__node_3 = eq(resp, OK);
  anchor read_phase__write_phase__node_4 = eq(cmd, WR) and rise(valid);
  anchor read_phase__write_phase__node_5 = eq(cmd, WR) and high(ready);

  window read_phase__write_phase__win_0 = between read_phase__write_phase__node_0 and read_phase__write_phase__point_1 [0:0];
  window read_phase__write_phase__win_1 = between read_phase__write_phase__point_1 and read_phase__write_phase__node_2 [0:0];
  window read_phase__write_phase__win_2 = between read_phase__write_phase__node_2 and read_phase__write_phase__node_3 [1:8];
  window read_phase__write_phase__win_3 = between read_phase__write_phase__node_4 and read_phase__write_phase__point_1 [0:0];
  window read_phase__write_phase__win_4 = between read_phase__write_phase__node_2 and read_phase__write_phase__node_5 [1:4];

  property read_phase:
    cmd == RD && valid |-> ##[1:8] resp == OK;
  property write_phase:
    cmd == WR && $rose(valid) |-> ##[1:4] cmd == WR && ready;
}