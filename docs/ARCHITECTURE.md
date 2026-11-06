# Architecture notes

## Request path

The gateway authenticates bearer credentials, converts supported Chat/Responses inputs, preserves reasoning/tool history and queues work behind one backend lock. A request selects a model ID. The manager verifies its configured artifacts, loads that backend if necessary, and proxies streaming output while observing client disconnects. Responses history is scoped to the credential owner and stored in local SQLite.

Linux and Windows remain separate implementations. Windows preserves its original session-aware design and pinned runtime patch. Linux's archived cache selection uses exact token prefixes and internal execution ancestry without a required caller session header.

## Prefix data and numerical state

The prefix index stores packed vocabulary token IDs. These are not embedding vectors. Full native snapshots hold execution state; periodic `RCPKV001` bundles store tokens plus partial target/draft/speculative state. Partial recurrent snapshots omit ordinary attention KV so a compatible resident KV prefix can remain on GPU.

Matching text/tokens is necessary but insufficient to mix state from different executions. Rewinding and recomputing a suffix creates a new epoch. An ancestor checkpoint is eligible only before its preserved-prefix cutoff. A logical family bounds durable checkpoint retention across these edits. Full snapshots carry the ancestry metadata needed for later restoration.

## Capture and storage

Capture waits for existing safe target/MTP boundaries; it does not split layer-major prefill batches solely to create checkpoints. GPU-to-host capture and initial staging are synchronous work. Checksumming, filesystem sync and commit use a bounded background queue. A queued checkpoint consumes a retention slot before its commit completes.

Native writers reserve space under a shared filesystem lock before writing. Combined accounting covers full snapshots, recurrent states, metadata and pending reservations across models. Old committed session groups can be evicted at capacity; abandoned writes are recovered after process exit. Idle expiry maintenance preserves active writes and does not unload a hot GPU session.

## Engine configuration

The Linux adapter reads local profile files containing engine arguments and artifact checksums. A compatible engine and matching libraries are required. Final checkpoint host sources are stored separately from the retained core sources because their ABIs differ. The native directories are source excerpts; top-level build orchestration and generated GPU artifacts are separate dependencies.
