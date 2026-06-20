#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass


DEFAULT_PORT = 8060
SERVER_SCRIPT = "live_narration_server.py"


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    command: str
    source: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detiene servidores locales de narracion FutBotMX.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Puerto a revisar. Default: {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra procesos candidatos, sin detenerlos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Usa SIGKILL si el proceso no termina con SIGTERM.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Segundos de espera despues de SIGTERM antes de reportar/forzar.",
    )
    args = parser.parse_args()

    processes = find_live_narration_processes(args.port)
    if not processes:
        print(f"No encontre servidores de narracion en puerto {args.port}.")
        return

    print("Procesos encontrados:")
    for process in processes:
        print(f"- pid={process.pid} source={process.source} command={process.command}")

    if args.dry_run:
        print("Dry run: no se detuvo ningun proceso.")
        return

    for process in processes:
        terminate_process(process.pid)

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        alive = [process for process in processes if is_process_alive(process.pid)]
        if not alive:
            print("Servidores detenidos correctamente.")
            return
        time.sleep(0.2)

    alive = [process for process in processes if is_process_alive(process.pid)]
    if not alive:
        print("Servidores detenidos correctamente.")
        return

    if args.force:
        for process in alive:
            kill_process(process.pid)
        print("Procesos restantes detenidos con SIGKILL.")
        return

    print("Algunos procesos siguen vivos. Reintenta con --force si quieres forzar:")
    for process in alive:
        print(f"- pid={process.pid} command={process.command}")


def find_live_narration_processes(port: int) -> list[ProcessInfo]:
    current_pid = os.getpid()
    by_pid: dict[int, ProcessInfo] = {}

    for pid, command in process_table():
        if pid == current_pid:
            continue
        if SERVER_SCRIPT in command:
            by_pid[pid] = ProcessInfo(pid=pid, command=command, source="script")

    for pid in pids_on_port(port):
        if pid == current_pid:
            continue
        command = command_for_pid(pid)
        existing = by_pid.get(pid)
        if existing:
            by_pid[pid] = ProcessInfo(pid=pid, command=existing.command, source="script+port")
        elif SERVER_SCRIPT in command:
            by_pid[pid] = ProcessInfo(pid=pid, command=command, source="port")

    return sorted(by_pid.values(), key=lambda process: process.pid)


def process_table() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            rows.append((int(pid_text), command.strip()))
        except ValueError:
            continue
    return rows


def pids_on_port(port: int) -> set[int]:
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        check=False,
        capture_output=True,
        text=True,
    )
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        try:
            pids.add(int(line.strip()))
        except ValueError:
            continue
    return pids


def command_for_pid(pid: int) -> str:
    for process_pid, command in process_table():
        if process_pid == pid:
            return command
    return "<unknown>"


def terminate_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"SIGTERM enviado a pid={pid}.")
    except ProcessLookupError:
        pass


def kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"SIGKILL enviado a pid={pid}.")
    except ProcessLookupError:
        pass


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


if __name__ == "__main__":
    main()
