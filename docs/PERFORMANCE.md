# Archived performance

These are selected measurements of the deployed Linux native builds. They are not a benchmark of the portable archive adapter. Reference hardware: Radeon AI PRO R9700 + GeForce RTX 3060, with 64 GB RAM.

All rows processed the full indicated input with no prefix reuse and generated 128 target-accepted output tokens. Model loading was outside the request timing. Measurements used the authenticated local service address with normal cache behavior; they do not represent a remote internet client's latency. The first request after each model load included cold execution/cache costs.

| Model | Input workload | Processed prefill tok/s | Accepted decode tok/s | Cold TTFT s |
|---|---|---:|---:|---:|
| `qwen3.8-27b` | 32K workload A | 1166.3 | 52.0 | 28.10 |
| `qwen3.8-27b` | 64K workload B | 1070.7 | 39.5 | 61.21 |
| `qwen3.8-27b` | 192K workload C | 782.9 | 34.0 | 251.13 |
| `qwen3.8-flash` | 32K workload A | 1092.6 | 40.9 | 29.99 |
| `qwen3.8-flash` | 64K workload B | 1112.4 | 32.9 | 58.91 |
| `qwen3.8-flash` | 192K workload C | 853.7 | 27.3 | 230.30 |

Each row is a single request, not an average or percentile. Accepted decode throughput uses native accepted-output counters. Cold TTFT includes prompt processing and is separate from cached-request latency. These text workloads do not characterize vision performance.

The benchmark corpus and raw logs are not included, so these figures provide reference measurements rather than a fully reproducible benchmark suite.

Windows measurements use different settings and workloads; see [Windows performance](../windows/docs/PERFORMANCE.md).
