"""Signal-name flavors and deterministic name allocation for generated scenarios."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class NameFlavor:
    """A pool of signal names mimicking a protocol family or domain."""

    name: str
    clock: str
    bit_signals: tuple[str, ...]
    bus_signals: tuple[str, ...]
    bus_widths: tuple[str, ...]
    naming_style: str  # snake_case|uppercase|protocol_like|short


FLAVORS: dict[str, NameFlavor] = {
    "generic": NameFlavor(
        name="generic",
        clock="clk",
        bit_signals=(
            "req",
            "ack",
            "valid",
            "ready",
            "done",
            "busy",
            "start",
            "stop",
            "grant",
            "resp_valid",
            "last",
        ),
        bus_signals=("addr", "data", "id", "tag", "len"),
        bus_widths=("8", "16", "32"),
        naming_style="snake_case",
    ),
    "axi_like": NameFlavor(
        name="axi_like",
        clock="ACLK",
        bit_signals=(
            "AWVALID",
            "AWREADY",
            "WVALID",
            "WREADY",
            "WLAST",
            "BVALID",
            "BREADY",
            "ARVALID",
            "ARREADY",
            "RVALID",
            "RREADY",
            "RLAST",
        ),
        bus_signals=("AWADDR", "WDATA", "WSTRB", "BRESP", "ARADDR", "RDATA", "RRESP"),
        bus_widths=("8", "16", "32"),
        naming_style="uppercase",
    ),
    "fifo": NameFlavor(
        name="fifo",
        clock="clk",
        bit_signals=("push", "pop", "full", "empty", "wreq", "rreq"),
        bus_signals=("wdata", "rdata", "level"),
        bus_widths=("8", "16"),
        naming_style="snake_case",
    ),
    "noc": NameFlavor(
        name="noc",
        clock="clk",
        bit_signals=("TX_VALID", "TX_READY", "RX_VALID", "RX_READY", "CR_VALID", "CR_READY"),
        bus_signals=("TX_HDR", "RX_DATA", "VC", "TX_DATA"),
        bus_widths=("8", "16", "32"),
        naming_style="protocol_like",
    ),
    "dma": NameFlavor(
        name="dma",
        clock="clk",
        bit_signals=("desc_valid", "desc_ready", "dma_start", "dma_done", "irq"),
        bus_signals=("desc_addr", "length", "status"),
        bus_widths=("16", "32"),
        naming_style="snake_case",
    ),
    "interrupt": NameFlavor(
        name="interrupt",
        clock="clk",
        bit_signals=("irq", "clear", "mask", "pending", "ack"),
        bus_signals=("cause", "status"),
        bus_widths=("8", "16"),
        naming_style="snake_case",
    ),
    "memory": NameFlavor(
        name="memory",
        clock="mclk",
        bit_signals=("cmd_valid", "cmd_ready", "rsp_valid", "rsp_ready"),
        bus_signals=("addr", "wdata", "rdata"),
        bus_widths=("8", "16", "32"),
        naming_style="snake_case",
    ),
}


PARAM_NAMES_BY_FLAVOR: dict[str, tuple[str, ...]] = {
    "generic": ("MAX_LAT", "READY_MAX", "RESP_MAX", "MAX_BEATS"),
    "axi_like": ("AW_READY_MAX", "W_READY_MAX", "BRESP_MAX", "RRESP_MAX"),
    "fifo": ("FIFO_MAX",),
    "noc": ("CR_MAX", "TX_MAX"),
    "dma": ("DESC_MAX", "DMA_MAX"),
    "interrupt": ("IRQ_MAX", "ACK_MAX"),
    "memory": ("CMD_MAX", "RSP_MAX"),
}


class NameAllocator:
    """Allocate distinct signal names from a flavor pool deterministically."""

    def __init__(self, flavor: NameFlavor, rng: random.Random) -> None:
        self.flavor = flavor
        self._rng = rng
        self._available_bits = list(flavor.bit_signals)
        self._available_bus = list(flavor.bus_signals)
        self._used: set[str] = set()
        rng.shuffle(self._available_bits)
        rng.shuffle(self._available_bus)

    def take_bit(self, hint: str | None = None) -> str:
        if hint and hint in self._available_bits:
            self._available_bits.remove(hint)
            self._used.add(hint)
            return hint
        if not self._available_bits:
            return self._synth("bit")
        name = self._available_bits.pop(0)
        self._used.add(name)
        return name

    def take_bus(self, hint: str | None = None) -> str:
        if hint and hint in self._available_bus:
            self._available_bus.remove(hint)
            self._used.add(hint)
            return hint
        if not self._available_bus:
            return self._synth("bus")
        name = self._available_bus.pop(0)
        self._used.add(name)
        return name

    def take_param(self, candidates: tuple[str, ...]) -> str:
        for name in candidates:
            if name not in self._used:
                self._used.add(name)
                return name
        return self._synth("PARAM")

    def take_bus_width(self) -> str:
        return self._rng.choice(self.flavor.bus_widths)

    def _synth(self, prefix: str) -> str:
        index = 0
        while True:
            candidate = f"{prefix}_{index}"
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate
            index += 1


def is_valid_identifier(name: str) -> bool:
    return bool(_IDENT_RE.fullmatch(name))
