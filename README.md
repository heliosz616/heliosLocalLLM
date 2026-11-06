# heliosLocalLLM

Source archive of two local LLM services: a Linux mixed-GPU deployment and an earlier Windows laptop/eGPU deployment. The repository preserves the serving layer, terminal controls, cache/state work, native source changes, tests and selected performance results.

**This is an archive, not a portable binary distribution.** Model weights, compiled engines, generated kernels, local settings, credentials, conversation history, KV snapshots and raw machine reports are excluded. The Linux gateway has been adapted to operator-supplied configuration; its complete native build has not been reproduced from this archive on a clean machine.

## Repository map

| Directory | Contents |
|---|---|
| [`linux/`](linux/README.md) | Linux gateway, CLI, configuration examples, templates and tests |
| [`linux/native/`](linux/native/README.md) | Retained core source areas, final checkpoint hosts and storage/policy tests |
| [`windows/`](windows/README.md) | Earlier Windows service, pinned runtime patch and deployment scripts |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Serving, exact-prefix caching and recurrent checkpoints |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Measurements and methodology |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | Upstream attribution and license locations |
| `tools/` | Source audit and index-only ZIP packaging |

## Archived services

| | Linux | Windows |
|---|---|---|
| Public model IDs | `qwen3.8-27b`, `qwen3.8-flash` | `swift-qwen3.8-27b` |
| Retained formats | Swift Q5_K_S; Flash UD-IQ4_XS | Swift IQ4_XS |
| Context configuration | 192K input, with output headroom | 128K |
| Attention cache | Q8 K/V | Q8 K/V |
| Speculation | Native MTP | Native MTP |
| Scheduling | One active inference stream; queued requests and model selection per request | One active inference stream; queued requests |
| Interfaces | Chat Completions, Responses, streaming, reasoning/tool-call handling | Chat-compatible gateway, streaming, reasoning/tool-call handling |

The Linux bare chat template adds no default system/persona instruction. Client-provided system messages and tool definitions are preserved. Reasoning labels are accepted, but labels other than `none` enable thinking without promising distinct compute budgets.

## Use or study the archive

- **Linux:** read [setup and controls](linux/README.md) before preparing a compatible native engine. `linux/configure.py` creates local settings and a new local API key; it does not download models or build an engine.
- **Windows:** follow the [Windows guide](windows/README.md) from the `windows/` directory. Its original pinned build workflow remains separate from Linux.
- **Native work:** see the [source inventory and build limits](linux/native/README.md). Checkpoint headers do not provide a complete inference engine by themselves.

Example client URL: `http://127.0.0.1:8080/v1` for the Linux gateway. Use your own configured address and locally generated key.

## Measured performance

The retained Linux results span approximately **783–1166 processed prefill tokens/s** and **27–52 target-accepted decode tokens/s** across the selected 32K/64K/192K workloads. These are historical measurements of the deployed native builds, **not measurements of a fresh installation of this archive**. See [the full table and methodology](docs/PERFORMANCE.md).

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream attribution and licenses.
