# Source packaging tools

`tools/audit_release.py` checks staged files and reachable Git history for credentials, local paths, private addresses, runtime payloads and unexpected file types. Findings report paths and rule names without printing matched values.

```bash
python3 tools/audit_release.py
```

An optional `--private-values` argument accepts a JSON array of values to detect. The file is an external, local-only input. Exact line-hash exceptions in `tools/privacy_allowlist.json` cover public upstream attribution and documentation syntax; they do not suppress known-value matching.

`tools/package_release.py` creates a ZIP from the Git index after the audit passes:

```bash
python3 tools/package_release.py --output ../heliosLocalLLM-source.zip
```

The ZIP contains staged blobs, normalized ZIP entry timestamps and Git file modes. Git metadata, ignored files and other working-directory files are excluded. ZIP entry timestamps are independent of Git commit timestamps.

These tools detect defined patterns; they do not provide a complete semantic review of arbitrary text.
