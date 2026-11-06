# Linux service archive

This directory contains the Linux serving layer and selected native sources. Public IDs are `qwen3.8-27b` (Swift Q5_K_S) and `qwen3.8-flash` (Flash UD-IQ4_XS). The archived hardware context is Radeon AI PRO R9700 + GeForce RTX 3060, with 64 GB RAM.

## Scope

The service includes a gateway, chat template, streaming cleanup, response history, cache quotas and retention logic. Engine profiles define executable paths, model artifacts and their checksums. Model weights, native executables and generated GPU artifacts are separate dependencies.

A compatible native engine is required. Its local extensions include exact-prefix SSD state, partial recurrent restoration and native MTP. A stock binary that ignores the checkpoint environment flags does not establish that these features work. The provided profile examples are configuration scaffolding, not validated GPU-placement recipes. Full native reconstruction and hardware validation remain operator work; see [`native/README.md`](native/README.md).

## Prepare local configuration

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r service/requirements-test.txt
.venv/bin/python configure.py
```

This creates ignored `config.local.json`, `engine-profiles.local.json` and private `state/serving/` data. Existing settings and keys are preserved. The default gateway bind address is loopback.

Before loading a model:

1. Provide the compatible native executables, shared libraries, original model weights and vision projectors outside Git.
2. Edit `engine-profiles.local.json`. Set the command argument arrays, loopback backend URLs, library paths and engine options for your installation. `${ROOT}` expands to this directory; environment placeholders can reference locally installed SDKs.
3. Fill every artifact SHA-256 from trusted build/model records. Cover the executable, loaded libraries, weights, projector and chat template. The gateway refuses missing or mismatched checksums; it does not automatically bless existing files.
4. Confirm placement, native MTP, Q8 K/V, FP32 recurrent state and full context with the actual engine. Run real generation, tool continuation, vision and long-context checks before relying on a new deployment.

Artifact checks in this portable adapter hash the configured artifacts before model activation. This adds work during model activation.

## Run and control

Foreground gateway:

```bash
.venv/bin/python service/gateway.py --config config.local.json
```

Optional user service (installs the unit without starting/enabling it):

```bash
.venv/bin/python configure.py --install-systemd
./llm-service start
./llm-service status
./llm-service load q5
./llm-service load flash
./llm-service info
./llm-service key
./llm-service history -n 20
./llm-service storage
./llm-service stop
```

The unit is `llm-service.service`. Startup prints discovered URLs. The CLI uses the configured port and local key. Keep one GPU service active at a time; independent copies do not coordinate GPU ownership automatically.

Clients send the base URL, model ID and `Authorization: Bearer <local-key>`. Select the model per request; it remains loaded until replacement or shutdown. Requests queue behind one physical inference stream. Logical cached sessions do not require a custom session header.

For LAN or a private overlay network, configure the gateway bind address locally and restrict access with your network/firewall policy. Native engines stay on loopback.

## Cache policy

- Exact tokenizer-ID prefix matching; no latent similarity lookup.
- Hot forward continuations retain resident GPU KV/state.
- Durable recurrent checkpoints: minimum 32,768-token spacing at existing safe batch boundaries, at most six per logical session family; pending commits occupy retention slots.
- Edits create a new execution epoch. Only compatible untouched ancestor prefixes may supply recurrent state.
- Periodic bundles contain recurrent/convolution/MTP state and token records, without ordinary attention KV.
- Full KV snapshots and recurrent bundles share a 64 GB decimal SSD budget and 24-hour expiry since last use.
- Current text-only periodic checkpointing leaves the existing vision path in place. Retained near-tail RAM checkpoints serve a different purpose from the six durable checkpoints.

Safe-boundary gaps can exceed 32K. Edits can therefore require substantial suffix replay; model/session replacement or restart may require a full SSD restore. Snapshot staging and suffix execution still cost time.

## Tests

```bash
.venv/bin/python -m pytest service/test_*.py -q
```

The policy, bundle and quota tests compile small C++ programs and require a C++17 compiler. Gateway tests use mock engines and temporary state. They do not validate physical GPU placement, long-context arithmetic or a complete native rebuild. Run the Windows tests in a separate pytest process to avoid same-named module collisions.
