# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``dns.upstream`` — upstream selection, probe, and rewrites.

No podman, no netns, no real sockets: the usability probe is injected as a
stub, and the socket-level functions are exercised through their pure
packet-building / response-parsing halves.
"""

import struct
import sys
from pathlib import Path
from unittest import mock

import pytest

from terok_shield.dns import upstream
from terok_shield.hooks import mode
from terok_shield.nft.constants import DNS_FALLBACKS
from terok_shield.resources import nft_hook
from terok_shield.state import StateBundle


# ── Env override: parsing + validation ─────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, None, id="unset"),
        pytest.param("", None, id="empty"),
        pytest.param("   ", None, id="whitespace"),
        pytest.param("1.1.1.1", "1.1.1.1", id="ipv4"),
        pytest.param(" 8.8.8.8 ", "8.8.8.8", id="ipv4-padded"),
        pytest.param("2606:4700:4700::1111", "2606:4700:4700::1111", id="ipv6"),
        pytest.param("2001:0db8::0001", "2001:db8::1", id="ipv6-canonicalized"),
        pytest.param("10.0.2.53", "10.0.2.53", id="private-rfc1918-allowed"),
        pytest.param("fdaa::53", "fdaa::53", id="private-ula-allowed"),
    ],
)
def test_parse_env_upstream_accepts_only_ip_literals(raw: str | None, expected: str | None) -> None:
    """A valid, non-floor IP literal is returned canonicalized."""
    assert upstream.parse_env_upstream(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("not-an-ip", id="garbage"),
        pytest.param("example.com", id="hostname"),
    ],
)
def test_parse_env_upstream_warns_and_returns_none_on_garbage(
    caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    """A malformed value warns and falls back — never raises, never returns it."""
    with caplog.at_level("WARNING"):
        assert upstream.parse_env_upstream(raw) is None
    assert upstream.ENV_UPSTREAM in caplog.text


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("127.0.0.1", id="ipv4-loopback"),
        pytest.param("127.0.0.53", id="ipv4-loopback-stub"),
        pytest.param("::1", id="ipv6-loopback"),
        pytest.param("169.254.1.1", id="ipv4-link-local-pasta"),
        pytest.param("169.254.169.254", id="ipv4-link-local-imds"),
        pytest.param("fe80::1", id="ipv6-link-local"),
    ],
)
def test_parse_env_upstream_rejects_loopback_and_link_local(
    caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    """A floor-range override is refused (warn) so it cannot reopen the hard-deny."""
    with caplog.at_level("WARNING"):
        assert upstream.parse_env_upstream(raw) is None
    assert upstream.ENV_UPSTREAM in caplog.text


# ── Env override: precedence in the pre_start selection path ──


@pytest.mark.parametrize("network_mode", ["pasta", "slirp4netns"])
def test_resolve_upstream_env_override_beats_mode(
    monkeypatch: pytest.MonkeyPatch, network_mode: str
) -> None:
    """TEROK_SHIELD_DNS_UPSTREAM wins over the mode mapping, flagged explicit."""
    monkeypatch.setenv(upstream.ENV_UPSTREAM, "9.9.9.9")
    assert mode._resolve_upstream_dns(network_mode) == ("9.9.9.9", True)


def test_resolve_upstream_falls_back_to_mode_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env var the mode mapping is used and flagged non-explicit."""
    monkeypatch.delenv(upstream.ENV_UPSTREAM, raising=False)
    assert mode._resolve_upstream_dns("pasta") == ("169.254.1.1", False)
    assert mode._resolve_upstream_dns("slirp4netns") == ("10.0.2.3", False)


def test_resolve_upstream_invalid_env_falls_back_to_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid override degrades to the mode mapping instead of crashing the hook."""
    monkeypatch.setenv(upstream.ENV_UPSTREAM, "definitely not an ip")
    assert mode._resolve_upstream_dns("pasta") == ("169.254.1.1", False)


def test_resolve_upstream_rejects_floor_range_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loopback/link-local override is refused and the mode mapping wins."""
    monkeypatch.setenv(upstream.ENV_UPSTREAM, "169.254.169.254")
    assert mode._resolve_upstream_dns("pasta") == ("169.254.1.1", False)


# ── Candidate ordering ─────────────────────────────────


def test_candidate_upstreams_defaults_to_the_preauthorized_fallbacks() -> None:
    """Without an explicit set, only the ruleset's pre-authorized fallbacks are offered."""
    assert upstream.candidate_upstreams("169.254.1.1") == list(DNS_FALLBACKS)


def test_candidate_upstreams_filters_primary_and_unusable() -> None:
    """The primary and unusable addresses (loopback/link-local/slirp) are dropped."""
    candidates = upstream.candidate_upstreams(
        "169.254.1.1",
        ["127.0.0.53", "192.0.2.1", "169.254.1.1", "10.0.2.3"],
    )
    assert candidates == ["192.0.2.1"]


def test_candidate_upstreams_deduplicates() -> None:
    """A repeated fallback is not emitted twice."""
    candidates = upstream.candidate_upstreams(
        "169.254.1.1", ["1.1.1.1", "1.1.1.1", "8.8.8.8"]
    )
    assert candidates == ["1.1.1.1", "8.8.8.8"]


# ── Selection with a stub probe ────────────────────────


def test_select_upstream_keeps_usable_primary() -> None:
    """A primary that probes usable is chosen without consulting fallbacks."""
    probed: list[str] = []

    def probe(server: str) -> bool:
        probed.append(server)
        return server == "169.254.1.1"

    chosen, reason = upstream.select_upstream("169.254.1.1", probe=probe, fallbacks=("1.1.1.1",))
    assert chosen == "169.254.1.1"
    assert probed == ["169.254.1.1"]
    assert "usable" in reason


def test_select_upstream_falls_back_to_preauthorized_public_resolver() -> None:
    """A REFUSED-style primary (probe false) hands off to the first usable fallback."""
    # Model the live failure: the primary answers, but with no answer records.
    answers = {"169.254.1.1": False, "1.1.1.1": True, "8.8.8.8": True}

    chosen, reason = upstream.select_upstream(
        "169.254.1.1",
        probe=lambda s: answers.get(s, False),
        fallbacks=DNS_FALLBACKS,
    )
    assert chosen == "1.1.1.1"
    assert "fallback" in reason


def test_select_upstream_tries_every_fallback_in_order() -> None:
    """The first unusable fallback is skipped; the next usable one is chosen."""
    usable = {"8.8.8.8"}

    chosen, reason = upstream.select_upstream(
        "169.254.1.1",
        probe=lambda s: s in usable,
        fallbacks=DNS_FALLBACKS,
    )
    assert chosen == "8.8.8.8"
    assert "fallback" in reason


def test_select_upstream_keeps_primary_when_nothing_usable() -> None:
    """No usable candidate — and no exception — keeps the primary untouched."""
    chosen, reason = upstream.select_upstream(
        "169.254.1.1", probe=lambda _s: False, fallbacks=DNS_FALLBACKS
    )
    assert chosen == "169.254.1.1"
    assert "no usable fallback" in reason


def test_select_upstream_explicit_skips_probe() -> None:
    """An explicit override is honored even when the probe would reject it."""
    calls: list[str] = []

    def probe(server: str) -> bool:
        calls.append(server)
        return False

    chosen, reason = upstream.select_upstream("9.9.9.9", probe=probe, explicit=True)
    assert chosen == "9.9.9.9"
    assert calls == []
    assert "explicit" in reason


def test_select_upstream_survives_a_broken_probe() -> None:
    """A probe that raises is treated as unusable — selection never propagates."""

    def probe(_server: str) -> bool:
        raise OSError("socket unavailable")

    chosen, reason = upstream.select_upstream(
        "169.254.1.1", probe=probe, fallbacks=DNS_FALLBACKS
    )
    assert chosen == "169.254.1.1"
    assert "no usable fallback" in reason


# ── DNS probe packet handling ──────────────────────────


def test_build_query_encodes_a_well_formed_question() -> None:
    """The query carries the transaction id, one question and recursion-desired."""
    tid, packet = upstream.build_query("example.com")
    assert 0 <= tid <= 0xFFFF
    assert packet[0:2] == tid.to_bytes(2, "big")
    assert packet[2:4] == b"\x01\x00"
    assert b"\x07example\x03com\x00" in packet


def test_response_is_answer_requires_noerror_with_an_answer() -> None:
    """Only NOERROR with ≥1 answer counts; REFUSED and empty responses do not."""
    tid = 0x1234
    refused = struct.pack("!HHHHHH", tid, 0x8185, 1, 0, 0, 0)  # QR=1 rcode=REFUSED
    empty = struct.pack("!HHHHHH", tid, 0x8180, 1, 0, 0, 0)  # QR=1 rcode=NOERROR
    answered = struct.pack("!HHHHHH", tid, 0x8180, 1, 1, 0, 0) + b"\x00" * 16
    assert upstream.response_is_answer(refused, tid) is False
    assert upstream.response_is_answer(empty, tid) is False
    assert upstream.response_is_answer(answered, tid) is True
    assert upstream.response_is_answer(answered, tid + 1) is False


def test_probe_upstream_false_on_bad_address() -> None:
    """A non-IP server string is unusable without touching the network."""
    assert upstream.probe_upstream("not-an-ip") is False


# ── Artifact rewrites ──────────────────────────────────

_DNSMASQ_CONF = (
    "# Generated by terok-shield (pid 1)\n"
    "listen-address=127.0.0.1\n"
    "port=53\n"
    "bind-interfaces\n"
    "no-resolv\n"
    "cache-size=0\n"
    "server=169.254.1.1\n"
    "pid-file=/run/dnsmasq.pid\n"
    "nftset=/github.com/4#inet#terok_shield#t40_project_allow_v4,"
    "6#inet#terok_shield#t40_project_allow_v6\n"
    "local=/blocked.example/\n"
    "server=/allowed.blocked.example/169.254.1.1\n"
)


def test_rewrite_dnsmasq_upstream_replaces_server_and_preserves_rest() -> None:
    """The plain ``server=`` line is repointed; nftset/local/other lines survive verbatim."""
    rewritten = upstream.rewrite_dnsmasq_upstream(_DNSMASQ_CONF, "1.1.1.1")
    lines = rewritten.splitlines()
    assert "server=1.1.1.1" in lines
    assert "server=/allowed.blocked.example/1.1.1.1" in lines  # punch-through repointed too
    assert "server=169.254.1.1" not in lines
    assert "nftset=/github.com/4#inet#terok_shield#t40_project_allow_v4," in rewritten
    assert "local=/blocked.example/" in lines
    assert "listen-address=127.0.0.1" in lines
    assert rewritten.endswith("\n")


def test_rewrite_dnsmasq_upstream_is_idempotent() -> None:
    """Rewriting to the same upstream leaves the text unchanged."""
    once = upstream.rewrite_dnsmasq_upstream(_DNSMASQ_CONF, "1.1.1.1")
    assert upstream.rewrite_dnsmasq_upstream(once, "1.1.1.1") == once


def test_rewrite_dnsmasq_upstream_leaves_address_less_domain_lines_verbatim() -> None:
    """Bare ``server=/dom`` and address-less ``server=/dom/`` are not mangled.

    dnsmasq's domain-without-address forms mean "use the standard servers";
    rewriting them to ``server=/<upstream>`` would change the semantics.
    """
    conf = "server=169.254.1.1\nserver=/bare.example\nserver=/dash.example/\n"
    out = upstream.rewrite_dnsmasq_upstream(conf, "1.1.1.1")
    lines = out.splitlines()
    assert "server=1.1.1.1" in lines
    assert "server=/bare.example" in lines
    assert "server=/dash.example/" in lines
    assert not any(line.startswith("server=/1.1.1.1") for line in lines)


def test_rewrite_resolv_conf_upstream_repoints_first_nameserver() -> None:
    """The nameserver line is repointed; ``options`` and comments survive."""
    text = "# generated\nnameserver 169.254.1.1\noptions ndots:0\n"
    rewritten = upstream.rewrite_resolv_conf_upstream(text, "1.1.1.1")
    assert "nameserver 1.1.1.1" in rewritten
    assert "options ndots:0" in rewritten
    assert "# generated" in rewritten
    assert "169.254.1.1" not in rewritten


# ── Runtime artifact rewrite (dnsmasq vs resolv tier) ──


def test_runtime_rewrite_fixes_dnsmasq_conf_for_dnsmasq_tier(tmp_path: Path) -> None:
    """A dnsmasq tier repoints dnsmasq.conf and leaves the bound resolv.conf alone."""
    bundle = StateBundle(tmp_path)
    bundle.ensure_dirs()
    bundle.dns_tier.write_text("dnsmasq-live\n")
    bundle.dnsmasq_conf.write_text(_DNSMASQ_CONF)
    bundle.resolv_conf.write_text("nameserver 127.0.0.1\noptions ndots:0\n")

    upstream._rewrite_artifacts(tmp_path, "1.1.1.1")

    assert "server=1.1.1.1" in bundle.dnsmasq_conf.read_text()
    assert bundle.resolv_conf.read_text() == "nameserver 127.0.0.1\noptions ndots:0\n"


def test_runtime_rewrite_fixes_resolv_conf_for_lookup_tier(tmp_path: Path) -> None:
    """A dnsmasq-less tier repoints the bound resolv.conf instead."""
    bundle = StateBundle(tmp_path)
    bundle.ensure_dirs()
    bundle.dns_tier.write_text("lookup\n")
    bundle.resolv_conf.write_text("nameserver 169.254.1.1\noptions ndots:0\n")

    upstream._rewrite_artifacts(tmp_path, "1.1.1.1")

    assert bundle.resolv_conf.read_text().startswith("nameserver 1.1.1.1\n")


def test_runtime_main_uses_first_fallback_for_mode_derived_primary_when_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unverifiable mode-derived forwarder is replaced by the first fallback.

    The create-time probe can run before the task netns has usable egress;
    keeping the pasta link-local address would then leave DNS broken, so the
    first pre-authorized fallback is used (unverified) instead.
    """
    bundle = StateBundle(tmp_path)
    bundle.ensure_dirs()
    bundle.upstream_dns.write_text("169.254.1.1\n")
    bundle.dns_tier.write_text("dnsmasq-live\n")
    bundle.dnsmasq_conf.write_text("server=169.254.1.1\n")
    monkeypatch.delenv(upstream.ENV_UPSTREAM, raising=False)
    monkeypatch.setattr(upstream, "probe_upstream", lambda *_a, **_k: False)

    assert upstream.runtime_main([str(tmp_path)]) == 0
    assert bundle.upstream_dns.read_text().strip() == DNS_FALLBACKS[0]
    assert f"server={DNS_FALLBACKS[0]}" in bundle.dnsmasq_conf.read_text()


def test_runtime_main_keeps_non_mode_primary_when_nothing_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-mode-derived primary is kept when nothing verifies (no blind switch)."""
    bundle = StateBundle(tmp_path)
    bundle.ensure_dirs()
    bundle.upstream_dns.write_text("198.51.100.1\n")
    bundle.dns_tier.write_text("dnsmasq-live\n")
    bundle.dnsmasq_conf.write_text("server=198.51.100.1\n")
    monkeypatch.delenv(upstream.ENV_UPSTREAM, raising=False)
    monkeypatch.setattr(upstream, "probe_upstream", lambda *_a, **_k: False)

    assert upstream.runtime_main([str(tmp_path)]) == 0
    assert bundle.upstream_dns.read_text().strip() == "198.51.100.1"
    assert "server=198.51.100.1" in bundle.dnsmasq_conf.read_text()


def test_runtime_main_honors_persisted_explicit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted explicit marker skips the probe even without the env var.

    Regression: the override must survive into the nsenter child via the state
    bundle, not ``os.environ`` (the child may not inherit it).
    """
    bundle = StateBundle(tmp_path)
    bundle.ensure_dirs()
    bundle.upstream_dns.write_text("169.254.1.1\n")
    bundle.upstream_explicit.write_text("1\n")
    bundle.dns_tier.write_text("dnsmasq-live\n")
    bundle.dnsmasq_conf.write_text(_DNSMASQ_CONF)
    monkeypatch.delenv(upstream.ENV_UPSTREAM, raising=False)
    probed: list[str] = []

    def probe(server: str, **_k: object) -> bool:
        probed.append(server)
        return True

    monkeypatch.setattr(upstream, "probe_upstream", probe)

    assert upstream.runtime_main([str(tmp_path)]) == 0
    assert probed == []  # explicit marker: no probe
    assert bundle.upstream_dns.read_text().strip() == "169.254.1.1"


def test_runtime_main_env_override_beats_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live env value still wins over the persisted marker and is honored."""
    bundle = StateBundle(tmp_path)
    bundle.ensure_dirs()
    bundle.upstream_dns.write_text("169.254.1.1\n")
    bundle.upstream_explicit.write_text("1\n")
    bundle.dns_tier.write_text("dnsmasq-live\n")
    bundle.dnsmasq_conf.write_text(_DNSMASQ_CONF)
    monkeypatch.setenv(upstream.ENV_UPSTREAM, "9.9.9.9")
    monkeypatch.setattr(upstream, "probe_upstream", lambda *_a, **_k: False)

    assert upstream.runtime_main([str(tmp_path)]) == 0
    assert bundle.upstream_dns.read_text().strip() == "9.9.9.9"
    assert "server=9.9.9.9" in bundle.dnsmasq_conf.read_text()


def test_runtime_main_persists_and_rewrites_on_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A usable fallback is written to upstream.dns and dnsmasq.conf."""
    bundle = StateBundle(tmp_path)
    bundle.ensure_dirs()
    bundle.upstream_dns.write_text("169.254.1.1\n")
    bundle.dns_tier.write_text("dnsmasq-live\n")
    bundle.dnsmasq_conf.write_text(_DNSMASQ_CONF)
    monkeypatch.delenv(upstream.ENV_UPSTREAM, raising=False)
    monkeypatch.setattr(upstream, "probe_upstream", lambda server, **_k: server == "1.1.1.1")

    assert upstream.runtime_main([str(tmp_path)]) == 0
    assert bundle.upstream_dns.read_text().strip() == "1.1.1.1"
    assert "server=1.1.1.1" in bundle.dnsmasq_conf.read_text()


# ── OCI-hook glue ──────────────────────────────────────


def test_hook_verify_upstream_skips_without_persisted_upstream(tmp_path: Path) -> None:
    """No ``upstream.dns`` (pre-feature bundle) means no nsenter probe at all."""
    with mock.patch("terok_shield.resources._oci_state.nsenter") as mock_ns:
        nft_hook._verify_upstream("42", tmp_path)
    mock_ns.assert_not_called()


def test_hook_verify_upstream_runs_module_under_nsenter(tmp_path: Path) -> None:
    """The persisted bundle triggers the selection module inside the container netns."""
    (tmp_path / "upstream.dns").write_text("169.254.1.1\n")
    with mock.patch("terok_shield.resources._oci_state.nsenter") as mock_ns:
        nft_hook._verify_upstream("42", tmp_path)
    mock_ns.assert_called_once_with(
        "42", sys.executable, "-I", "-m", "terok_shield.dns.upstream", str(tmp_path)
    )


def test_hook_verify_upstream_never_propagates_failure(tmp_path: Path) -> None:
    """A broken nsenter/probe must be swallowed — container start proceeds."""
    (tmp_path / "upstream.dns").write_text("169.254.1.1\n")
    with mock.patch(
        "terok_shield.resources._oci_state.nsenter",
        side_effect=RuntimeError("nsenter unavailable"),
    ):
        nft_hook._verify_upstream("42", tmp_path)  # must not raise
