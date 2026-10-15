# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Choosing a *usable* DNS upstream for a container's actual netns.

``pre_start`` picks the upstream from the host-global network mode
(``pasta`` → ``169.254.1.1``, ``slirp4netns`` → ``10.0.2.3``).  That mode is
derived from ``podman info`` and describes the *default* rootless setup — it
can disagree with the network namespace a given task actually runs in (an
explicit ``--network``, an unrestricted/"dangerous" task, a re-created
container, …).  When it does, the mode-derived address is a foreign resolver
inside that netns: it answers, but returns ``REFUSED`` for every recursive
name, so the container resolves nothing while the host resolves fine through
the same address.

Two layers feed a usable pick:

* an explicit operator/project override,
  ``TEROK_SHIELD_DNS_UPSTREAM`` — an IP literal that wins over the mode
  mapping and is honored even when the runtime probe below fails; and
* a netns-verified fallback: a functional UDP DNS query (``NOERROR`` with at
  least one answer) run *inside the container's netns* before dnsmasq
  starts.  A mere reachability check would wrongly pass the live failure mode
  (the address answers, just with ``REFUSED``), so the probe speaks DNS.  The
  fallback set is exactly the public resolvers the ruleset preamble
  pre-authorizes (``nft.constants.DNS_FALLBACKS``) — a site-specific host
  resolver is not pre-authorized and would be firewalled (or, if added, could
  punch the private floor), so it must come through the explicit override
  instead.

This module holds the pure, testable half: env parsing, candidate ordering,
selection with an injected probe, and the on-disk config rewrites.  The one
piece that cannot be unit-tested here is the netns execution — the OCI hook
re-runs this module with ``python3 -m terok_shield.dns.upstream <state_dir>``
via ``nsenter`` (see ``resources/nft_hook.py``), so the probe and the rewrite
run inside the container netns.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
import secrets
import socket
import struct
from collections.abc import Callable, Sequence
from pathlib import Path

from ..nft.constants import DNS_FALLBACKS, SLIRP4NETNS_DNS
from ..state import StateBundle

logger = logging.getLogger(__name__)

ENV_UPSTREAM = "TEROK_SHIELD_DNS_UPSTREAM"
"""Explicit upstream-DNS override — an IP literal, honored over the mode mapping."""

PROBE_NAME = "example.com"
"""Well-known recursive name the usability probe asks for."""

PROBE_TIMEOUT_S = 2.0
"""Per-candidate socket timeout for the usability probe."""


# ── Env override ───────────────────────────────────────


def parse_env_upstream(raw: str | None) -> str | None:
    """Validate the ``TEROK_SHIELD_DNS_UPSTREAM`` value, or ``None`` when unusable.

    An empty value means "unset".  A malformed value is a warning, never a
    crash — the hook must keep working and fall back to the mode mapping.

    Loopback (``127.0.0.0/8``) and link-local (``169.254.0.0/16``, incl. the
    cloud-metadata IMDS ``169.254.169.254``; IPv6 ``fe80::/10``) literals are
    rejected: the override is baked into the ruleset preamble as a ``:53``
    accept *above* the hard-deny/private floors, so those ranges must not be
    reachable through it.  These are the same ranges ``_is_usable_candidate``
    refuses as fallbacks.  Private RFC1918/ULA addresses stay allowed — a
    site-specific internal resolver is the legitimate use case.  The returned
    address is canonicalized so equality checks against persisted state are
    reliable regardless of the input notation.
    """
    if raw is None or not raw.strip():
        return None
    candidate = raw.strip()
    try:
        addr = ipaddress.ip_address(candidate)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r (not an IP literal); using the mode-derived upstream",
            ENV_UPSTREAM,
            raw,
        )
        return None
    if addr.is_loopback or addr.is_link_local:
        logger.warning(
            "Ignoring %s=%r (loopback/link-local cannot be an upstream override); "
            "using the mode-derived upstream",
            ENV_UPSTREAM,
            raw,
        )
        return None
    return str(addr)


# ── Candidate ordering ─────────────────────────────────


def _is_usable_candidate(ip: str) -> bool:
    """True when *ip* is worth probing as a fallback upstream.

    Rejects the addresses that cannot be a generic recursive resolver:
    loopback stubs (``127.0.0.0/8``), link-local space including pasta's
    ``169.254.1.1`` (``169.254.0.0/16``), multicast/unspecified, and the
    slirp4netns default ``10.0.2.3`` (a foreign-but-answering address in a
    non-slirp netns — the same failure mode as pasta's link-local).
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return False
    return str(addr) != SLIRP4NETNS_DNS


def candidate_upstreams(
    primary: str,
    fallbacks: Sequence[str] = DNS_FALLBACKS,
) -> list[str]:
    """Ordered, deduplicated fallback candidates from the pre-authorized set.

    Only the public resolvers the ruleset preamble pre-authorizes are offered:
    anything outside that set is firewalled under the default-deny posture, so
    probing it would only waste a query.  *primary* is excluded (it is probed
    separately), as is anything :func:`_is_usable_candidate` rejects.
    """
    seen = {primary}
    ordered: list[str] = []
    for ip in fallbacks:
        if ip in seen or not _is_usable_candidate(ip):
            continue
        seen.add(ip)
        ordered.append(ip)
    return ordered


# ── Selection ──────────────────────────────────────────


def _probe_ok(probe: Callable[[str], bool], server: str) -> bool:
    """Run *probe* on *server*, treating any exception as "unusable"."""
    try:
        return bool(probe(server))
    except Exception:  # noqa: BLE001 — a broken probe must never crash the hook
        return False


def select_upstream(
    primary: str,
    *,
    probe: Callable[[str], bool],
    fallbacks: Sequence[str] = DNS_FALLBACKS,
    explicit: bool = False,
) -> tuple[str, str]:
    """Pick the upstream to use, verifying usability with *probe*.

    An *explicit* override is honored unconditionally — the operator's word
    beats a flaky probe (the probe runs in a throwaway netns state and may
    fail for reasons that do not apply to steady-state operation).

    Otherwise *primary* is probed first; on failure the ordered candidates
    from :func:`candidate_upstreams` are tried.  If none is usable — or the
    probe machinery itself is broken — the *primary* is kept unchanged so the
    container never ends up without its resolver artifact.  Returns
    ``(chosen, reason)`` for logging.
    """
    if explicit:
        return primary, "explicit override — probe skipped"
    if _probe_ok(probe, primary):
        return primary, "primary upstream is usable"
    for candidate in candidate_upstreams(primary, fallbacks):
        if _probe_ok(probe, candidate):
            return candidate, f"primary unusable; fallback {candidate} is usable"
    return primary, "no usable fallback; keeping primary upstream"


# ── Usability probe (raw UDP DNS query) ────────────────


def build_query(name: str) -> tuple[int, bytes]:
    """A minimal recursive DNS query for an ``A`` record, plus its transaction id.

    Dependency-free on purpose: the hook ships to hosts where ``dig`` may be
    absent, and a subprocess lookup introduces an unwanted tool dependency.
    """
    tid = secrets.randbelow(0x10000)
    flags = 0x0100  # standard query, recursion desired
    question = b"".join(bytes([len(label)]) + label.encode() for label in name.split("."))
    packet = (
        struct.pack("!HHHHHH", tid, flags, 1, 0, 0, 0)
        + question
        + b"\x00"
        + struct.pack("!HH", 1, 1)  # QTYPE A, QCLASS IN
    )
    return tid, packet


def response_is_answer(data: bytes, tid: int) -> bool:
    """True when *data* is a DNS reply for *tid* with ``NOERROR`` and ≥1 answer."""
    if len(data) < 12:
        return False
    rid, flags, _qd, ancount, _ns, _ar = struct.unpack("!HHHHHH", data[:12])
    if rid != tid or not flags & 0x8000:  # response bit
        return False
    return (flags & 0x000F) == 0 and ancount >= 1  # NOERROR with an answer


def probe_upstream(server: str, *, name: str = PROBE_NAME, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Ask *server* for *name*; True only on a real answer, never just reachability.

    Runs in whatever network namespace the calling process is in, so the OCI
    hook invokes this module under ``nsenter`` to test from the container's
    netns.  ``REFUSED``/``SERVFAIL``/timeout/no-answer all read as unusable.
    """
    try:
        family = socket.AF_INET6 if ipaddress.ip_address(server).version == 6 else socket.AF_INET
    except ValueError:
        return False
    tid, packet = build_query(name)
    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        sock.sendto(packet, (server, 53))
        data, _peer = sock.recvfrom(4096)
    except OSError:
        return False
    finally:
        sock.close()
    return response_is_answer(data, tid)


# ── Artifact rewrites ──────────────────────────────────


def rewrite_dnsmasq_upstream(conf_text: str, upstream: str) -> str:
    """Point every ``server=`` directive in *conf_text* at *upstream*.

    Rewrites the plain ``server=<ip>`` upstream line and the embedded upstream
    of domain-scoped punch-throughs (``server=/dom/<ip>``); every other line —
    ``nftset=``, ``local=``, ``listen-address=`` — is preserved byte for byte.
    """
    lines: list[str] = []
    for line in conf_text.splitlines():
        if not line.startswith("server="):
            lines.append(line)
            continue
        value = line[len("server=") :]
        if value.startswith("/"):
            # Domain-scoped only when a second "/" delimits a non-empty
            # address component; a bare ``server=/dom`` or address-less
            # ``server=/dom/`` is left verbatim (dnsmasq's "standard servers"
            # form) rather than being mangled into ``server=/<upstream>``.
            prefix, sep, address = value.rpartition("/")
            if "/" in value[1:] and address:
                lines.append(f"server={prefix}{sep}{upstream}")
            else:
                lines.append(line)
        else:
            lines.append(f"server={upstream}")
    return "\n".join(lines) + ("\n" if conf_text.endswith("\n") else "")


def rewrite_resolv_conf_upstream(text: str, upstream: str) -> str:
    """Point the first ``nameserver`` line of *text* at *upstream*.

    Used on the tiers that run no dnsmasq (lookup/getent), where the bound
    ``resolv.conf`` names the forwarder directly.  Other lines (``options``,
    comments) are preserved.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.split()[:1] == ["nameserver"]:
            lines[i] = f"nameserver {upstream}"
            break
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


# ── Runtime entry point (executed under nsenter) ───────


def _rewrite_artifacts(state_dir: Path, chosen: str) -> None:
    """Rewrite the dnsmasq conf or the bound resolv.conf for *state_dir*.

    dnsmasq tiers forward through dnsmasq, so its ``server=`` line is the one
    that matters and the bound ``resolv.conf`` (which points at the dnsmasq
    bind address) must stay untouched.  The lookup/getent tiers have no
    dnsmasq, so their resolv.conf is rewritten instead.
    """
    bundle = StateBundle(state_dir)
    tier = bundle.read_dns_tier()
    runs_dnsmasq = tier.runs_dnsmasq if tier is not None else bundle.dnsmasq_conf.is_file()
    if runs_dnsmasq:
        if bundle.dnsmasq_conf.is_file():
            bundle.dnsmasq_conf.write_text(
                rewrite_dnsmasq_upstream(bundle.dnsmasq_conf.read_text(), chosen)
            )
    elif bundle.resolv_conf.is_file():
        bundle.resolv_conf.write_text(
            rewrite_resolv_conf_upstream(bundle.resolv_conf.read_text(), chosen)
        )


def _log_decision(state_dir: Path, primary: str, chosen: str, reason: str) -> None:
    """Record the decision on stderr and in the operator-visible hook log."""
    msg = f"terok-shield dns-upstream: primary={primary} chosen={chosen} ({reason})"
    logger.info(msg)
    try:
        with (state_dir / "hook-error.log").open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")
    except OSError:
        pass


def runtime_main(argv: Sequence[str] | None = None) -> int:
    """Netns-side entry point: verify the persisted upstream and fix it in place.

    Invoked by the OCI hook as ``python3 -m terok_shield.dns.upstream
    <state_dir>`` inside ``nsenter``.  Always returns 0: a missing bundle, an
    unreadable state dir, or an unbroken primary must never fail the
    container start — the worst outcome is "keep the primary".
    """
    parser = argparse.ArgumentParser(prog="terok_shield.dns.upstream")
    parser.add_argument("state_dir", type=Path)
    args = parser.parse_args(argv)

    bundle = StateBundle(args.state_dir)
    if not bundle.upstream_dns.is_file():
        return 0
    primary = bundle.upstream_dns.read_text().strip()
    if not primary:
        return 0

    # An explicit override is honored even if the child does not inherit the
    # hook's environment: the persisted marker (written by ``pre_start``) is
    # authoritative.  A live env value still wins over the marker so a caller
    # can point the netns-side check at a different resolver.
    env_value = parse_env_upstream(os.environ.get(ENV_UPSTREAM))
    explicit = env_value is not None or bundle.upstream_explicit.is_file()
    effective = env_value if env_value is not None else primary
    chosen, reason = select_upstream(
        effective,
        probe=probe_upstream,
        explicit=explicit,
    )
    if chosen != primary:
        _rewrite_artifacts(args.state_dir, chosen)
        bundle.upstream_dns.write_text(f"{chosen}\n")
    _log_decision(args.state_dir, primary, chosen, reason)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(runtime_main())
