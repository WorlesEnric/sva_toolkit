diagram td_single_response_generic_td_000014 {
  clock posedge clk;
  ticks 16;

  param MAX_LAT;

  lane req: bit = 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
  lane busy: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane ack: bit = 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
  lane len: bus[16] = x 7E 7E 7E 7E x x x x x x x x x D0 x;
  lane addr: bus[16] = x FF FF FF FF x x x x x x x x x 10 x;
  lane id: bus[8] = x 33 33 33 33 x x x x x x x x x 02 x;
  lane data: bus[8] = x x x 02 01 A5 x x x x x x x x x x;

  anchor trigger = rise(req);
  anchor response = high(busy) and high(ack);

  window response_window = between trigger and response [1:MAX_LAT];

  show stable(len) from trigger until response;
  show stable(addr) in response_window;
  show stable(id) in response_window;
  show low(busy) before trigger;
  show change(data) at response;
}
