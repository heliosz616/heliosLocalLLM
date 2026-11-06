# Retained native source areas

| Path | Scope |
|---|---|
| `flash-core/` | Retained Flash core, GGML, common helpers, vision sources and vendor dependencies |
| `q5-core/` | Retained Q5 core, GGML, common helpers, vision sources and vendor dependencies |
| `hosts/flash/server-context.cpp` | Final Flash checkpoint host source (v12) |
| `hosts/q5/server-context.cpp` | Final Q5 checkpoint host source (v10) |
| `checkpoints/` | Shared policy, bundle/quota/storage headers, integration fragment and small native tests |
| `provenance.json` | Scope and reproduction limits |

This is a source archive, not a complete standalone CMake checkout. Top-level engine build orchestration, private manifests, generated binaries/artifacts and selected upstream non-source areas are omitted. Some retained component CMake files refer to files outside this excerpt. Do not treat copying a header onto an arbitrary upstream revision as a supported build procedure.

The host sources were separated from the core versions because their ABIs differ. A compatible engine checkout and matching headers/libraries are required. The excerpt does not supply a complete native build or GPU validation workflow. The storage/policy unit tests are independently runnable through the Linux Python test suite.

The optional FlyDSL adapter reads its compiled artifact path from `LLM_SERVICE_FLYDSL_ARTIFACT`. When unset, that optional path is unavailable. The generated artifact and its compiler/runtime are separate dependencies.

The source includes experimental paths. Runtime settings determine which paths execute. Model formats are Q5_K_S and UD-IQ4_XS; recurrent state uses FP32 and attention K/V uses Q8.

Upstream source notices are retained. See [`THIRD_PARTY.md`](../../THIRD_PARTY.md) and the component license files before redistribution.
