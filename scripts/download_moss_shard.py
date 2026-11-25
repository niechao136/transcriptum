#!/usr/bin/env python3
"""Resumable, retrying downloader for the MOSS safetensors shard.

Works around hf-mirror throttling by using plain HTTP Range requests and
appending to a local partial file. Verifies SHA256 when complete.
"""
import hashlib
import os
import sys
import time
import urllib.request

URL = ("https://hf-mirror.com/OpenMOSS-Team/MOSS-Transcribe-Diarize/"
       "resolve/main/model-00000-of-00001.safetensors")
TARGET = "/home/nechao/models/MOSS-Transcribe-Diarize/model-00000-of-00001.safetensors"
EXPECTED_SIZE = 1817113576
EXPECTED_SHA = "9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4"
PARTIAL = TARGET + ".partial"
CHUNK = 4 * 1024 * 1024


# Pinned IPv4 addresses to bypass WSL2's flaky DNS resolver.
PINNED_IPS = {
    "hf-mirror.com": "104.250.143.18",
    # Verified serving fresh signed URLs; the other resolved IPs returned 401.
    "cas-bridge.xethub.hf.co": "18.245.31.85",
}


def force_ipv4():
    import socket
    orig = socket.getaddrinfo
    def ipv4(host, port, family=0, type=0, proto=0, flags=0):
        if host in PINNED_IPS:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PINNED_IPS[host], port))]
        return orig(host, port, socket.AF_INET, type, proto, flags)
    socket.getaddrinfo = ipv4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_range(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}",
                                               "User-Agent": "moss-deploy/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        if r.status != 206:
            raise RuntimeError(f"expected 206, got {r.status}")
        return r.read()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    force_ipv4()
    # Reuse existing hf-xet partial if present and smaller than target.
    if not os.path.exists(PARTIAL):
        xet = None
        cache_dir = os.path.expanduser(
            "~/models/MOSS-Transcribe-Diarize/.cache/huggingface/download")
        if os.path.isdir(cache_dir):
            for name in os.listdir(cache_dir):
                if name.endswith(".incomplete"):
                    p = os.path.join(cache_dir, name)
                    if os.path.getsize(p) < EXPECTED_SIZE:
                        xet = p
                        break
        if xet:
            os.replace(xet, PARTIAL)
            log(f"reused xet partial: {os.path.getsize(PARTIAL)} bytes")

    pos = os.path.getsize(PARTIAL) if os.path.exists(PARTIAL) else 0
    log(f"resuming from {pos} / {EXPECTED_SIZE} bytes")
    if pos >= EXPECTED_SIZE:
        log("already complete, verifying")
    else:
        with open(PARTIAL, "ab") as out:
            while pos < EXPECTED_SIZE:
                end = min(pos + CHUNK, EXPECTED_SIZE) - 1
                for attempt in range(8):
                    try:
                        data = http_range(pos, end)
                        if not data:
                            raise RuntimeError("empty response")
                        out.write(data)
                        out.flush()
                        pos += len(data)
                        pct = 100.0 * pos / EXPECTED_SIZE
                        log(f"{pos}/{EXPECTED_SIZE} ({pct:.1f}%) +{len(data)}B")
                        break
                    except Exception as e:
                        log(f"retry {attempt+1}/8 at {pos}: {e}")
                        time.sleep(min(2 ** attempt, 30))
                else:
                    log("giving up after 8 retries")
                    return 1
    got = sha256_file(PARTIAL)
    if got != EXPECTED_SHA:
        log(f"SHA256 MISMATCH: got {got}, expected {EXPECTED_SHA}")
        return 2
    os.replace(PARTIAL, TARGET)
    log(f"DONE: {TARGET} sha256 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
