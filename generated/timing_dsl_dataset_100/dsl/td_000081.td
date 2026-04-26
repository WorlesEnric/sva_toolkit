diagram td_single_response_interrupt_td_000081 {
  clock posedge clk;
  ticks 6;

  lane cause: bus[8] = D0 A1 A1 x x x;
  lane ack: bit = 0 1 1 0 0 0;
  lane clear: bit = 0 0 0 0 1 0;
  lane status: bus[16] = 00 FF FF x x x;
  lane pending: bit = 0 0 0 0 0 0;

  anchor trigger = high(ack);
  anchor response = stable(cause);

  window response_window = between trigger and response [1:1];

  show high(ack) from trigger until response;
  show low(clear) before trigger;
  show status = FF in response_window;
  show low(pending) after response;
}
