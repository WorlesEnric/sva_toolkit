from __future__ import annotations

import pytest

from sva_toolkit.describe import SVADTranslator
from sva_toolkit.describe.translator import SUPPORTED_SYSTEM_FUNCTIONS


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


REQUIRED_SYSTEM_FUNCTIONS = {
    "$past",
    "$sampled",
    "$rewind",
    "$past_gclk",
    "$future_gclk",
    "$assertcontrol",
    "$asserton",
    "$assertoff",
    "$assertpassoff",
    "$assertfailoff",
    "$assertpassoncontrol",
    "$assertfailoncontrol",
    "$assertnonvacuouson",
    "$assertvacuousoff",
    "$error",
    "$fatal",
    "$warning",
    "$info",
}


@pytest.mark.parametrize(
    ("call_text", "expected_fragment"),
    [
        ("$past(req)", "the previous value of the request (req) signal"),
        ("$sampled(req)", "the sampled value of the request (req) signal"),
        ("$rewind(req)", "the rewound sampled value of the request (req) signal"),
        ("$past_gclk(req, 2)", "the global-clock value of the request (req) signal from 2 ticks ago"),
        ("$future_gclk(req, 3)", "the global-clock value of the request (req) signal 3 ticks in the future"),
        ("$assertcontrol(1, req)", "assertion control is updated with arguments 1, the request (req) signal"),
        ("$asserton(req)", "assertion checking is enabled with arguments the request (req) signal"),
        ("$assertoff(req)", "assertion checking is disabled with arguments the request (req) signal"),
        ("$assertpassoff(req)", "assertion pass-action reporting is disabled with arguments the request (req) signal"),
        ("$assertfailoff(req)", "assertion fail-action reporting is disabled with arguments the request (req) signal"),
        ("$assertpassoncontrol(req)", "assertion pass-action control is enabled with arguments the request (req) signal"),
        ("$assertfailoncontrol(req)", "assertion fail-action control is enabled with arguments the request (req) signal"),
        ("$assertnonvacuouson(req)", "non-vacuous assertion reporting is enabled with arguments the request (req) signal"),
        ("$assertvacuousoff(req)", "vacuous assertion reporting is disabled with arguments the request (req) signal"),
        ('$error("bad req")', 'an error is reported with arguments "bad req"'),
        ('$fatal(2, "bad req")', 'a fatal error is reported with arguments 2, "bad req"'),
        ('$warning("bad req")', 'a warning is reported with arguments "bad req"'),
        ('$info("bad req")', 'an informational message is reported with arguments "bad req"'),
    ],
)
def test_translator_expands_required_system_function_templates(
    call_text: str,
    expected_fragment: str,
) -> None:
    translator = SVADTranslator()

    rendered = translator.translate(f"assert property (@(posedge clk) {call_text});")

    assert expected_fragment in rendered


def test_translator_registers_required_system_function_templates() -> None:
    assert REQUIRED_SYSTEM_FUNCTIONS <= SUPPORTED_SYSTEM_FUNCTIONS
