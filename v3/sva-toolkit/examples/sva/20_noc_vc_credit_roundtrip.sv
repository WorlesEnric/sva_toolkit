property tx_wait(int TX_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(TX_VALID) |-> ##[0:TX_READY_MAX] (TX_VALID && TX_READY);
endproperty

property rx_start(int RX_START_MAX);
  @(posedge clk) disable iff (!rst_n)
    (TX_VALID && TX_READY) |-> ##[1:RX_START_MAX] $rose(RX_VALID);
endproperty

property rx_first_wait(int RX_READY_MAX);
  @(posedge clk) disable iff (!rst_n)
    $rose(RX_VALID) |-> ##[0:RX_READY_MAX] (RX_VALID && RX_READY);
endproperty

property rx_phase;
  @(posedge clk) disable iff (!rst_n)
    $rose(RX_VALID) |-> ##[0:8] (RX_VALID && RX_READY && RX_TAIL);
endproperty

property cr_latency(int CREDIT_LAT_MAX);
  @(posedge clk) disable iff (!rst_n)
    (RX_VALID && RX_READY && RX_TAIL) |-> ##[0:CREDIT_LAT_MAX] $rose(CR_VALID);
endproperty

property cr_wait;
  @(posedge clk) disable iff (!rst_n)
    $rose(CR_VALID) |-> ##[0:1] (CR_VALID && CR_READY);
endproperty

property high_TX_VALID_from_tx_rise_until_tx_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(TX_VALID) |-> TX_VALID until_with (TX_VALID && TX_READY);
endproperty

property stable_TX_VC_from_tx_rise_until_tx_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(TX_VALID) |-> $stable(TX_VC) until_with (TX_VALID && TX_READY);
endproperty

property stable_TX_DEST_from_tx_rise_until_tx_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(TX_VALID) |-> $stable(TX_DEST) until_with (TX_VALID && TX_READY);
endproperty

property stable_TX_HDR_from_tx_rise_until_tx_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(TX_VALID) |-> $stable(TX_HDR) until_with (TX_VALID && TX_READY);
endproperty

property stable_TX_CLASS_from_tx_rise_until_tx_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(TX_VALID) |-> $stable(TX_CLASS) until_with (TX_VALID && TX_READY);
endproperty

property high_RX_VALID_from_rx_rise_until_rx_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(RX_VALID) |-> RX_VALID until_with (RX_VALID && RX_READY && RX_TAIL);
endproperty

property stable_RX_DATA_from_rx_rise_until_rx_first;
  @(posedge clk) disable iff (!rst_n)
    $rose(RX_VALID) |-> $stable(RX_DATA) until_with (RX_VALID && RX_READY);
endproperty

property stable_RX_VC_from_rx_rise_until_rx_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(RX_VALID) |-> $stable(RX_VC) until_with (RX_VALID && RX_READY && RX_TAIL);
endproperty

property stable_RX_CLASS_from_rx_rise_until_rx_last;
  @(posedge clk) disable iff (!rst_n)
    $rose(RX_VALID) |-> $stable(RX_CLASS) until_with (RX_VALID && RX_READY && RX_TAIL);
endproperty

property high_CR_VALID_from_cr_rise_until_cr_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CR_VALID) |-> CR_VALID until_with (CR_VALID && CR_READY);
endproperty

property stable_CR_VC_from_cr_rise_until_cr_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CR_VALID) |-> $stable(CR_VC) until_with (CR_VALID && CR_READY);
endproperty

property stable_CR_COUNT_from_cr_rise_until_cr_hs;
  @(posedge clk) disable iff (!rst_n)
    $rose(CR_VALID) |-> $stable(CR_COUNT) until_with (CR_VALID && CR_READY);
endproperty

property low_RX_VALID_before_tx_hs;
  @(posedge clk) disable iff (!rst_n)
    !RX_VALID until (TX_VALID && TX_READY);
endproperty
