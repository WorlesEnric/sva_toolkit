"""Curated signal presets for representative SVA generation."""

from __future__ import annotations


def _dedupe(*groups: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for signal in group:
            if signal not in seen:
                seen.add(signal)
                ordered.append(signal)
    return ordered


HANDSHAKE_SIGNALS = [
    "req",
    "ack",
    "gnt",
    "grant",
    "request",
    "valid",
    "ready",
    "accept",
    "stall",
    "busy",
]

FIFO_SIGNALS = [
    "push",
    "pop",
    "full",
    "empty",
    "almost_full",
    "almost_empty",
    "fifo_wr_en",
    "fifo_rd_en",
    "fifo_full",
    "fifo_empty",
]

AXI_SIGNALS = [
    "awvalid",
    "awready",
    "awaddr",
    "awlen",
    "wvalid",
    "wready",
    "wdata",
    "wlast",
    "bvalid",
    "bready",
    "arvalid",
    "arready",
    "araddr",
    "arlen",
    "rvalid",
    "rready",
    "rdata",
    "rlast",
]

CLOCK_SIGNALS = [
    "clk",
    "clock",
    "ref_clk",
    "sys_clk",
    "core_clk",
    "bus_clk",
    "mem_clk",
    "axi_clk",
    "clk_i",
    "clk_o",
    "clk_en",
    "clk_main",
    "clk_fast",
    "clk_slow",
    "clk_div2",
    "clk_div4",
    "clk_phy",
    "clk_pcie",
    "clk_usb",
    "clk_ddr",
    "clk_cpu",
    "clk_dma",
    "clk_uart",
    "clk_spi",
    "clk_i2c",
    "clk_can",
    "clk_video",
    "clk_audio",
    "clk_sensor",
    "clk_mac",
]

RESET_SIGNALS = [
    "rst",
    "rst_n",
    "reset",
    "reset_n",
    "soft_reset",
    "hard_reset",
    "async_rst",
    "async_rst_n",
    "sync_rst",
    "sync_rst_n",
    "system_reset",
    "reset_done",
]

CONTROL_SIGNALS = [
    "enable",
    "disable",
    "start",
    "stop",
    "done",
    "init_done",
    "clear",
    "flush",
    "load",
    "store",
    "commit",
    "rollback",
    "issue",
    "retire",
    "pause",
    "resume",
    "sleep",
    "wake",
    "mode",
    "state",
    "state_next",
    "state_valid",
    "fsm_idle",
    "fsm_busy",
    "fsm_error",
]

DATA_SIGNALS = [
    "data_in",
    "data_out",
    "data_valid",
    "data_ready",
    "data_en",
    "data_bus",
    "data_mask",
    "data_last",
    "addr",
    "addr_valid",
    "write_data",
    "read_data",
    "byte_enable",
    "word_count",
    "burst_len",
    "payload_valid",
    "header_valid",
    "packet_start",
    "packet_end",
    "packet_valid",
    "frame_start",
    "frame_end",
    "frame_valid",
    "transfer_complete",
]

MEMORY_SIGNALS = [
    "mem_req",
    "mem_ack",
    "mem_addr",
    "mem_wdata",
    "mem_rdata",
    "mem_write",
    "mem_read",
    "mem_ready",
    "cache_hit",
    "cache_miss",
    "linefill_done",
    "evict_req",
    "evict_done",
    "refresh_req",
    "refresh_ack",
    "bank_open",
    "bank_conflict",
]

POWER_SIGNALS = [
    "power_good",
    "power_ok",
    "power_down",
    "power_down_req",
    "power_down_ack",
    "sleep_req",
    "sleep_ack",
    "wakeup_req",
    "wakeup_ack",
    "pll_lock",
    "pll_locked",
    "clock_stable",
]

ERROR_SIGNALS = [
    "error",
    "error_flag",
    "err_flag",
    "crc_error",
    "ecc_error",
    "timeout",
    "overflow",
    "underflow",
    "parity_error",
    "protocol_error",
    "fault_detected",
    "alert",
]

INTERRUPT_SIGNALS = [
    "interrupt",
    "interrupt_pending",
    "int_pending",
    "int_ack",
    "irq",
    "irq_ack",
    "nmi",
    "trap",
    "exception",
    "exception_ack",
]

SECURITY_SIGNALS = [
    "secure_boot",
    "auth_valid",
    "key_valid",
    "decrypt_done",
    "encrypt_done",
    "tamper_detected",
    "privileged_mode",
    "debug_mode",
    "lock",
    "unlock",
]

DEFAULT_SIGNALS = _dedupe(
    HANDSHAKE_SIGNALS,
    FIFO_SIGNALS,
    AXI_SIGNALS,
    CLOCK_SIGNALS,
    RESET_SIGNALS,
    CONTROL_SIGNALS,
    DATA_SIGNALS,
    MEMORY_SIGNALS,
    POWER_SIGNALS,
    ERROR_SIGNALS,
    INTERRUPT_SIGNALS,
    SECURITY_SIGNALS,
)
