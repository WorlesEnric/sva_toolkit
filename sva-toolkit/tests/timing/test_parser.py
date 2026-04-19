from __future__ import annotations

import pytest

from sva_toolkit.timing.core.conditions import Predicate
from sva_toolkit.timing.core.scenario import WindowBoundKind
from sva_toolkit.timing.errors import TimingDslError
from sva_toolkit.timing.frontend.parser import parse_diagram


def test_parse_diagram_accepts_minimal_valid_diagram() -> None:
    document = parse_diagram(
        """
        diagram test {
          clock posedge clk;
          lane req: bit = 0 0 1 1;
          ticks 4;
        }
        """
    )

    assert document.name == "test"
    assert document.clocking.edge == "posedge"
    assert document.clocking.signal == "clk"
    assert document.ticks == 4
    assert len(document.signals) == 1
    assert document.signals[0].name == "req"
    assert document.signals[0].samples == ("0", "0", "1", "1")


def test_parse_diagram_parses_anchors_and_windows() -> None:
    document = parse_diagram(
        """
        diagram flow {
          clock posedge clk;
          lane req: bit = 0 1 1 1;
          lane gnt: bit = 0 0 1 1;
          anchor start = rise(req);
          anchor done = high(gnt);
          window response = between start and done [1:2];
          ticks 4;
        }
        """
    )

    assert tuple(anchor.name for anchor in document.anchors) == ("start", "done")
    assert document.anchor_map["start"].condition.predicate == Predicate(op="rise", signal="req")
    assert document.window_map["response"].start_anchor == "start"
    assert document.window_map["response"].end_anchor == "done"
    assert document.window_map["response"].bound.kind == WindowBoundKind.RANGE
    assert document.window_map["response"].bound.min_delay == "1"
    assert document.window_map["response"].bound.max_delay == "2"


def test_parse_diagram_raises_for_invalid_diagram() -> None:
    with pytest.raises(TimingDslError, match="missing clock declaration"):
        parse_diagram(
            """
            diagram broken {
              lane req: bit = 0 0 1 1;
              ticks 4;
            }
            """
        )


def test_parse_diagram_preserves_property_notes_from_comments() -> None:
    document = parse_diagram(
        """
        diagram extracted {
          clock posedge clk;

          lane req: bit;
          anchor unsupported = 1'b1;

          property p0 [unsupported]:
            req |-> ack;
          // p0: DAG compilation failed: unsupported feature
        }
        """
    )

    assert document.properties[0].status.value == "unsupported"
    assert document.properties[0].notes == ("DAG compilation failed: unsupported feature",)
