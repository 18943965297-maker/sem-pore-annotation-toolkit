"""Fail closed on unexpected tracked files or common private content patterns."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    allowed = set((ROOT / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines())
    tracked = set(git("ls-files", "-z").decode().split("\0")) - {""}
    if tracked != allowed:
        raise SystemExit(f"Allowlist mismatch: extra={sorted(tracked-allowed)}, missing={sorted(allowed-tracked)}")
    issues = []
    patterns = {
        "private path": r"(?i)(?:(?<![A-Za-z])[A-Z]:[\\/]|/root/|/home/)",
        "private host": r"(?i)(?:seetacloud|autodl|root@)",
        "access token": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})",
        "private key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        "assigned credential": r'''(?i)(?:password|api_key|access_token)\s*[:=]\s*["'][^"']+["']''',
    }
    total = 0
    for name in sorted(tracked):
        if Path(name).suffix.lower() not in {".py", ".md", ".txt", ".yml", ".cff"} and name not in {".gitignore", ".gitattributes"}:
            issues.append(f"Disallowed extension: {name}")
        data = git("show", f":{name}")  # inspect staged bytes, not just the working tree
        total += len(data)
        if len(data) > 400_000 or b"\0" in data:
            issues.append(f"Binary/oversized: {name}")
        content = data.decode("utf-8")
        if name != "tools/audit_public_files.py":
            for label, pattern in patterns.items():
                if re.search(pattern, content):
                    issues.append(f"{label}: {name}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"PASS: {len(tracked)} allowlisted text files, {total:,} bytes; no images, weights or archives tracked.")


if __name__ == "__main__":
    main()
