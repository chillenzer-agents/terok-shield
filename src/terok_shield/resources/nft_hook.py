#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
"""OCI hook: apply pre-generated terok-shield nft ruleset.

Applies ``ruleset.nft`` (written by ``pre_start()``) and optionally
starts dnsmasq if ``dnsmasq.conf`` is present in the state directory.
Gateway addresses are baked into the ruleset at generation time — no
runtime ``/proc`` discovery needed.

Stdlib-only by design, except for a sibling-module import of
``_oci_state`` shipped to the same hooks directory at install time.
The OCI runtime executes setup's Python in isolated mode; no installed
terok packages are imported.

The hook is invoked by crun, which runs inside podman's rootless user
namespace (``NS_ROOTLESS``).  Inside ``NS_ROOTLESS`` ``os.getuid() ==
0`` and ``CAP_NET_ADMIN`` is already available, so ``nsenter -n -t
<pid>`` is used directly.  When run from a normal shell
(``NS_INIT``, uid != 0), ``podman unshare nsenter -n -t <pid>`` is
used instead — see ``_oci_state.nsenter`` for the dispatch.
"""

import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path

# Isolated Python omits the script directory from sys.path. Add only our
# setup-owned directory so the copied ballast is importable, without importing
# installed terok packages or accepting ambient PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _oci_state  # noqa: E402 — sys.path bootstrap precedes import

# WORKAROUND(selinux-nft-stdin): nft runs as iptables_t (SELinux
# domain), which cannot read files in data_home_t (~/.local/share/…).
# Piping the ruleset via stdin (``-f -``) bypasses the file-read
# restriction — the hook process (not nft) opens the file.
# Remove when: SELinux policy grants nft read access to user state
# dirs, or state_dir moves to an nft-accessible location.

_RULESET_NAME = "ruleset.nft"
_CONTAINER_ID_NAME = "container.id"


def main() -> int:
    """OCI hook entry point: dispatch by stage."""
    _oci_state.bootstrap_env()
    stage = sys.argv[1] if len(sys.argv) > 1 else "createRuntime"
    try:
        oci = json.load(sys.stdin)
    except ValueError as exc:
        _oci_state.log(f"terok-shield hook: bad OCI state: {exc}")
        return 1

    sd = _oci_state.state_dir_from_oci(oci)
    if sd is None:
        return 1

    # All subsequent errors go to <state_dir>/hook-error.log so they
    # survive even when the OCI runtime does not forward the hook's
    # stderr.
    log_path = sd / "hook-error.log"

    try:
        return _nft_main(oci, sd, stage, log_path)
    except Exception as exc:  # noqa: BLE001
        _oci_state.log(f"terok-shield hook: {exc}", log_path)
        return 1


def _nft_main(oci: dict, sd: Path, stage: str, log_path: Path) -> int:
    """Apply the nft ruleset and (optionally) dnsmasq at the right stage."""
    # poststop cleanup must run regardless of bundle-version — a
    # container that was started before a terok-shield upgrade still
    # needs its dnsmasq reaped on stop.
    if stage == "poststop":
        _poststop(sd)
        return 0
    if stage != "createRuntime":
        _oci_state.log(f"terok-shield hook: unknown stage {stage!r}", log_path)
        return 1

    # Fail fast on any version mismatch — deliberately no compatibility
    # window and no migration: the annotation is frozen at container
    # creation, so a mismatch means this container was prepared by a
    # different terok-shield generation, and the remedy is a fresh
    # container (re-create the task; the task workspace rides its mounts).
    ann = oci.get("annotations") or {}
    ver = ann.get(_oci_state.ANN_VERSION, "")
    if not ver or str(ver) != str(_oci_state.BUNDLE_VERSION):
        _oci_state.log(
            f"terok-shield hook: bundle version {ver!r} != {_oci_state.BUNDLE_VERSION} — this "
            "container was prepared by a different terok-shield generation. Re-create the "
            "task (a fresh container gets a current shield bundle); if the installed hooks "
            "are older than the package, re-run 'terok-shield setup'.",
            log_path,
        )
        return 1

    pid = str(oci.get("pid") or "")
    if not pid:
        _oci_state.log("terok-shield hook: missing pid in OCI state", log_path)
        return 1

    # Persist container ID for D-Bus bridge bus-name derivation.  The
    # OCI state includes "id" (full 64-char hex); store the short form.
    container_id = str(oci.get("id") or "")
    if container_id:
        (sd / _CONTAINER_ID_NAME).write_text(container_id[:12] + "\n")

    _createruntime(pid, sd)
    return 0


def _createruntime(pid: str, sd: Path) -> None:
    """Apply the pre-generated ruleset and optionally start dnsmasq."""
    # Verify the target PID's network namespace file exists before
    # invoking nsenter.  Use stat() rather than exists(): Path.exists()
    # silently swallows PermissionError in Python 3.14+, whereas stat()
    # reliably raises it.  PermissionError means the file is present
    # but we can't read it (non-root caller) — that's fine, proceed.
    # Any other OSError (FileNotFoundError, etc.) means the PID is gone.
    ns_net = Path(f"/proc/{pid}/ns/net")
    try:
        ns_net.stat()
    except PermissionError:
        pass
    except OSError:
        raise RuntimeError(f"network namespace file missing for pid {pid}: {ns_net}")

    ruleset = sd / _RULESET_NAME
    if not ruleset.exists():
        raise RuntimeError(f"ruleset.nft not found: {ruleset}")
    nft = _oci_state.find_nft()
    # WORKAROUND(selinux-nft-stdin): pipe ruleset via stdin; see header.
    _oci_state.nsenter(pid, nft, "-f", "-", stdin=ruleset.read_text())

    # Gateway addresses are baked into ruleset.nft by pre_start() — no
    # /proc discovery or dynamic nft set population needed.

    # Verify (and if needed rewrite) the upstream DNS before dnsmasq starts.
    _verify_upstream(pid, sd)

    _start_container_dnsmasq(pid, sd)


def _verify_upstream(pid: str, sd: Path) -> None:
    """Re-check the persisted upstream DNS from inside the container's netns.

    ``pre_start`` chose the upstream from the host-global network mode, which
    can disagree with the namespace this task actually runs in; the mismatch
    yields a foreign-but-answering resolver that returns ``REFUSED`` for every
    recursive name (container resolves nothing).  Re-running the selection in
    the real netns — via ``nsenter``, the same mechanism the ruleset/dnsmasq
    launches use — lets a functional probe pick a usable fallback and rewrite
    the artifacts before dnsmasq reads them.

    Entirely best-effort: a missing bundle, a missing installed package, an
    unreachable ``nsenter``, or any probe failure leaves the primary upstream
    untouched and logs.  The container must never come up without its
    resolver artifact because this extra hardening step could not run.

    An already-installed hook script from before this change (an upgrade that
    did not re-run ``terok setup``) does not call this at all, so the runtime
    verification is skipped silently and the fallback accepts the new
    ``pre_start`` adds stay inert.
    """
    if not (sd / _oci_state.UPSTREAM_DNS_FILE_NAME).is_file():
        return  # pre-dates the upstream file — nothing to verify
    try:
        # The child is the package's own netns-side helper; run it as
        # isolated as this hook (`-I`) so it imports terok_shield from the
        # installation's site-packages without honouring ambient PYTHONPATH.
        _oci_state.nsenter(
            pid,
            sys.executable,
            "-I",
            "-m",
            "terok_shield.dns.upstream",
            str(sd),
        )
    except Exception as exc:  # noqa: BLE001 — hardening must not fail the start
        _oci_state.log(
            f"terok-shield hook: DNS upstream verification skipped ({exc}) — "
            "keeping the persisted upstream",
            sd / "hook-error.log",
        )


def _container_netns_inode(pid: str) -> int | None:
    """Network-namespace inode of the target container's init process.

    The hook itself runs on the host (or in podman's rootless userns), *not*
    inside the container's netns — it enters that netns with ``nsenter``.  The
    container's netns identity is therefore ``/proc/<pid>/ns/net`` for the OCI
    ``pid``, which is what a live dnsmasq must share.  ``None`` when
    unreadable, which disables the netns guard rather than producing a false
    mismatch.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    return _oci_state.netns_inode(pid_int)


def _start_container_dnsmasq(pid: str, sd: Path) -> None:
    """Launch dnsmasq in the container's netns and verify it started.

    No-op if ``dnsmasq.conf`` is absent (pre_start did not enable
    dnsmasq).  resolv.conf is pre-written by pre_start and bind-mounted
    :ro, so DNS already points to the configured bind address before
    the hook runs.

    For non-loopback binds (krun runtime), assigns the listen address
    to ``lo`` inside the netns before launching dnsmasq — ``bind-interfaces``
    requires the address to be present on an interface.  Loopback
    (``127.0.0.0/8``) is already on ``lo`` by kernel default, so the
    add step is skipped.

    Clears a stale PID file, then verifies the new PID file and process
    identity after start.
    """
    dnsmasq_conf = sd / _oci_state.DNSMASQ_CONF_FILE_NAME
    if not dnsmasq_conf.exists():
        return

    pid_file = sd / _oci_state.DNSMASQ_PID_FILE_NAME
    # Idempotent: if a dnsmasq is already running against our conf **in the
    # target container's netns** (because the hook fired twice — restart,
    # re-dispatch, sibling hook re-entry, etc.), don't try to bind port 53
    # again.  The netns guard matters on re-creation: the PID/conf identity
    # alone would match an orphan left in the previous container's netns by a
    # failed poststop reap, which would wrongly suppress this launch and leave
    # the new container with no resolver.
    if _our_dnsmasq_alive(pid_file, sd, _container_netns_inode(pid)):
        return

    # Add the dnsmasq listen address to ``lo`` for non-loopback binds.
    # Reads from dnsmasq.conf so the runtime info doesn't need a second
    # state file or a new annotation — the conf is the source of truth.
    bind = _oci_state.parse_dnsmasq_listen_address(dnsmasq_conf)
    if bind and not bind.startswith("127."):
        _oci_state.add_local_ip(pid, bind)

    # Remove stale PID file so the post-launch check is not fooled by a
    # leftover from a previous run (mirrors dnsmasq.launch()).
    try:
        pid_file.unlink()
    except OSError:
        pass

    binary = _oci_state.find_dnsmasq(sd)
    (sd / _oci_state.DNSMASQ_BIN_FILE_NAME).write_text(binary + "\n")
    _oci_state.nsenter(pid, binary, f"--conf-file={dnsmasq_conf}")

    try:
        dnsmasq_pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        raise RuntimeError(
            f"dnsmasq started but PID file not written at {pid_file}. "
            "The container's DNS may not be functional."
        )
    if not _oci_state.is_our_dnsmasq(dnsmasq_pid, sd):
        raise RuntimeError(
            f"dnsmasq PID {dnsmasq_pid} is not the expected process. "
            "The container's DNS may not be functional."
        )


def _poststop(sd: Path) -> None:
    """Send SIGTERM to the per-container dnsmasq process (best-effort).

    Verifies PID identity against ``/proc/{pid}/cmdline`` before
    signalling to avoid hitting an unrelated process when the original
    dnsmasq PID is recycled after container stop.
    """
    pid_file = sd / _oci_state.DNSMASQ_PID_FILE_NAME
    if not pid_file.exists():
        return
    try:
        pid_int = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return
    if not _oci_state.is_our_dnsmasq(pid_int, sd):
        _oci_state.log(
            f"poststop: stale dnsmasq pid file (pid {pid_int} recycled or gone) — removed",
            sd / "hook-error.log",
        )
        with contextlib.suppress(OSError):
            pid_file.unlink()
        return
    # Escalate and report: a silently-leaked dnsmasq per container stop is
    # invisible to the operator, and EPERM here (userns mismatch) would
    # otherwise vanish into a bare except.  Signal through ``podman unshare``
    # because the dnsmasq is owned by the container's mapped root (a host
    # subuid), which the operator uid cannot kill directly.
    try:
        _oci_state.signal_dnsmasq(pid_int, signal.SIGTERM)
    except OSError as exc:
        _oci_state.log(f"poststop: SIGTERM dnsmasq[{pid_int}] failed: {exc}", sd / "hook-error.log")
        return
    for _ in range(20):  # up to 2s for dnsmasq to exit on its own
        if not _oci_state.is_our_dnsmasq(pid_int, sd):
            return
        time.sleep(0.1)
    try:
        _oci_state.signal_dnsmasq(pid_int, signal.SIGKILL)
        _oci_state.log(
            f"poststop: dnsmasq[{pid_int}] survived SIGTERM — sent SIGKILL",
            sd / "hook-error.log",
        )
    except OSError as exc:
        _oci_state.log(f"poststop: SIGKILL dnsmasq[{pid_int}] failed: {exc}", sd / "hook-error.log")


def _our_dnsmasq_alive(pid_file: Path, sd: Path, expected_netns: int | None = None) -> bool:
    """``True`` when ``pid_file`` names a live dnsmasq of this container.

    *expected_netns* (a network-namespace inode) additionally requires the
    process to share the target container's netns, so an orphan matching only
    on argv/conf — left in a stale netns by a previous container that reused
    this state dir — is correctly reported as *not* alive.
    """
    try:
        pid_int = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    return _oci_state.is_our_dnsmasq(pid_int, sd, expected_netns=expected_netns)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
