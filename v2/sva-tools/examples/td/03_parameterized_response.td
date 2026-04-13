// Example 3: Parameterized timing with disable-iff
// Demonstrates: parameters, disable iff, parameterized window bounds

diagram parameterized_response {
  clock posedge clk;
  disable iff !rst_n;

  param MIN_LAT;
  param MAX_LAT;

  lane req:   bit;
  lane grant: bit;

  anchor trigger = rise(req);
  anchor resp   = rise(grant);

  window latency = between trigger and resp [MIN_LAT:MAX_LAT];

  property grant_response:
    req |-> ##[MIN_LAT:MAX_LAT] grant;
}
