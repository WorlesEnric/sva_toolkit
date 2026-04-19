from __future__ import annotations

from sva_toolkit.describe import SVADTranslator


def test_translator_accepts_raw_property_text_and_renders_sections() -> None:
    translator = SVADTranslator()

    rendered = translator.translate(
        """
        property req_ack;
            @(posedge clk) req |-> ##[1:3] ack;
        endproperty
        """
    )

    assert "Relevant Signals" in rendered
    assert "Check Condition" in rendered
    assert "Expected Results" in rendered
    assert "`clk`" in rendered
    assert "`req`" in rendered
    assert "`ack`" in rendered
    assert "Exp_0" in rendered
    assert "Exp_1" in rendered


def test_translator_documents_disable_condition_and_builtin_functions() -> None:
    translator = SVADTranslator()

    rendered = translator.translate(
        """
        assert property (@(posedge clk) disable iff (!rst_n)
            $rose(req) |=> ##1 $stable(data));
        """
    )

    assert "disable" in rendered.lower() or "scope" in rendered.lower()
    assert "rst_n" in rendered
    assert "Sys_0" in rendered
    assert "Sys_1" in rendered
    assert "$rose" not in rendered
    assert "$stable" not in rendered

