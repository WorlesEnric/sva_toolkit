"""Regression tests for the grammar-based timing DSL parser."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from sva_toolkit.timing.errors import TimingSyntaxError
from sva_toolkit.timing.frontend.parser import parse_diagram


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "td"
LEGACY_DOCUMENT_HASHES = {
    "01_simple_handshake.td": "736e39b32e41ad7bcc29f05eda3b2d49470a2af2da71ec7e6eae51592fb6cf33",
    "02_data_stable.td": "8f83065ce01171c79b35e71c62f63ff58cbb183f63ecfd006181904a61f04042",
    "03_parameterized_response.td": "ee651942b8bbc54c781768ac57edb6a26a47e50cba6effe0c7e2058afb8dec4e",
    "04_hold_until_ready.td": "f1a49ef10e83ee0035691232d6cd4142c13d079e0cb6148c1331ea3bbeb04fad",
    "05_legacy_rules.td": "e6842669b244cfa008dff37c10f0c4e3713675c9735b31580fd754a0fae67c73",
    "06_bus_protocol.td": "fa924bee02bd48d4c5e0f91567d7fc53805c1ef39540fdbb248f1b634d454356",
    "07_symbolic_pipeline.td": "5e52cdff2457654f79436f4889fd46c71a5e60f346225328287b4f42c29415fc",
    "08_multi_phase_transaction.td": "b82e0f6e84cab9b513530e5e4cc908788aa190661b63a4cc8bcda812fabe3104",
    "09_fifo_interface.td": "e93acc0eeff96235b353fe45ce94006af207167cf3024dc5b0e67c5e4b5b557f",
    "10_axi4_write_channel.td": "c04a2c689c90821b2c12b8550146fa57129b8c028d060dea2a073aefb19807bb",
    "11_emit_sva_bridge.td": "9d4dd7691ce3743ec7c3a47a8c8854bf62b0a015f324ad13210db60292c1278f",
    "12_extract_sva_bridge.td": "69b8deb20e3d3ba9f9f92ad099555375c4e80168471b4db9e3a31bc51440518d",
    "13_axi4_read_channel.td": "baa0f4a0aed303c6bf5799af7e01467cd9b0ff21f56ed4a749bab61c8edcad73",
    "14_apb_error_irq_flow.td": "3940e70b999e17ba4b08aed1928f22acc227d9c54d44068ebeaf8f54c21a5552",
    "15_tilelink_get_grantdata.td": "c5bb00f004081f225d9e99108edaf31204684881d85b0d5b6646d00a5c29dc4a",
    "16_dma_descriptor_completion.td": "a5a2a52397b3a6b29e51d06efdd7a63db4a379a36d225342c8d307abc063deed",
    "17_cacheline_refill.td": "f922c1745857b0c2fd25d2c94cdde542ca20980083d34c914de7c963848cfe3e",
    "18_chi_readshared_datreturn.td": "b76ce1af400c0aecb81f5e58134300d58e3c25508de4587be53e2830a944cd28",
    "19_pcie_cfg_read_completion.td": "948af0f3c6a638aa2e311fe7861d6d6b3b0fcb71f1fddb9da65a1d1f66f6edb1",
    "20_noc_vc_credit_roundtrip.td": "3145e30a7021bf7f16b022af36c22803e03ab133450e72160947ea30121e02aa",
    "21_usb3_ep0_control_read.td": "3bed8dd52eb1bb9ea6d430bc16796d8b744553a27dd29ff8ae0c4c14ba59982a",
    "22_lpddr_refresh_blocked_read.td": "cab1da5dea59de4be370474a492960cba01fffd47c3ac964f31e8af2e66af4dc",
}


@pytest.mark.parametrize("example_name, expected_hash", sorted(LEGACY_DOCUMENT_HASHES.items()))
def test_examples_match_legacy_parser_output(example_name: str, expected_hash: str) -> None:
    document = parse_diagram((EXAMPLES_DIR / example_name).read_text())
    assert _document_hash(document) == expected_hash


def test_parse_diagram_tolerates_hash_comments_and_multiline_declarations() -> None:
    document = parse_diagram(
        """
        # leading shell-style comment
        diagram multiline_comments {
          clock posedge clk; # trailing hash comment
          ticks 4; # comment after scalar statement

          lane req:
            bit =
              0 0
              1 1; # wrapped samples
          lane ack:
            bit =
              0 0
              0 1;

          anchor req_rise =
            rise(req); # wrapped anchor
          window response =
            between req_rise and req_rise
            [0:0];

          property req_to_ack:
            req
            |-> ##[0:0]
            ack; # wrapped property body
        }
        """
    )

    assert document.name == "multiline_comments"
    assert document.ticks == 4
    assert document.signal_map["req"].samples == ("0", "0", "1", "1")
    assert document.window_map["response"].bound.min_delay == "0"
    assert document.properties[0].body == "req |-> ##[0:0] ack"


def test_parse_diagram_reports_clean_line_column_for_dangling_parenthesis() -> None:
    with pytest.raises(TimingSyntaxError, match=r"expected '\)' to close '\(' at \d+:\d+") as exc_info:
        parse_diagram(
            """
            diagram broken {
              clock posedge clk;
              ticks 2;
              lane req: bit = 0 1;
              anchor start = rise(req;
            }
            """
        )

    assert exc_info.value.line > 0
    assert exc_info.value.column > 0


def _document_hash(document) -> str:
    payload = json.dumps(asdict(document), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
