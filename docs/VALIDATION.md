# Testing

Run the Python suites separately because the Linux and Windows modules have overlapping names:

```bash
python3 -m pytest linux/service/test_*.py -q
python3 -m pytest windows/qwen27b/tests -q
```

Install the respective service dependencies first. Linux checkpoint, bundle and quota tests also require a C++17 compiler.

The suites cover gateway behavior, templates, streaming, tool continuation, cache policy, state storage and configuration validation. Backend lifecycle tests use mock engines. They do not establish native GPU correctness, performance or a complete engine rebuild.

Changes to native engines require real generation, checkpoint restoration, vision and long-context validation. See [PERFORMANCE.md](PERFORMANCE.md) for reference measurements.
