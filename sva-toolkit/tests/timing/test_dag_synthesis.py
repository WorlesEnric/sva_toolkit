from sva_toolkit.timing.bridge.from_sva import extract_sva_scenario, bundle_sva_scenarios
from sva_toolkit.timing.bridge.emit_sva import synthesize_sva_from_diagram

def test_multi_sva_bundling_temporal_unification():
    """Verify that multiple SVAs with a shared trigger are unified in a single diagram timeline."""
    sva1 = "assert property (@(posedge clk) req |-> ##1 ack);"
    sva2 = "assert property (@(posedge clk) req |-> ##2 done);"
    
    doc1 = extract_sva_scenario(sva1, name="sva1")
    doc2 = extract_sva_scenario(sva2, name="sva2")
    
    # Bundle them with a lower threshold to ensure they are merged despite 33% overlap
    bundled = bundle_sva_scenarios([doc1, doc2], overlap_threshold=0.3)
    
    assert len(bundled) == 1
    doc = bundled[0]
    
    # Count anchors that represent 'req'
    from sva_toolkit.timing.core.conditions import collect_signals
    req_anchors = [a for a in doc.anchors if "req" in collect_signals(a.condition)]
    assert len(req_anchors) == 1, "req anchors should be merged across SVAs"
    
    # Check windows
    assert len(doc.windows) >= 2

def test_sva_synthesis_from_diagram():
    """Verify that we can synthesize SVA properties from a diagram's temporal structure."""
    from sva_toolkit.timing.frontend.parser import parse_diagram
    
    dsl = """
    diagram my_test {
      clock posedge clk;
      ticks 4;
      lane req: bit = 0 1 0 0;
      lane ack: bit = 0 0 1 0;
      
      anchor a_req = rise(req);
      anchor a_ack = rise(ack);
      
      window w1 = between a_req and a_ack [1:1];
    }
    """
    doc = parse_diagram(dsl)
    
    # Synthesize SVA
    sva_code = synthesize_sva_from_diagram(doc)
    
    assert "property" in sva_code
    assert "req" in sva_code
    # |=> is non-overlapped ##1
    assert "|=>" in sva_code or "##1" in sva_code or "|->" in sva_code
    assert "ack" in sva_code
