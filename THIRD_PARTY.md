# Third-party sources and attribution

- **llama.cpp / GGML:** <https://github.com/ggml-org/llama.cpp>. Native excerpts and host code derive from this project and related local changes. MIT license copies are in `linux/native/flash-core/LICENSE`, `linux/native/q5-core/LICENSE` and `windows/licenses/llama.cpp-MIT.txt`. Preserve per-file notices. Linux binary-to-source reconstruction is not claimed; Windows's pinned upstream revision remains in its build script.
- **Vendored dependencies:** JSON for Modern C++, cpp-httplib, xxHash, miniaudio, stb and other small dependencies retain their embedded notices and available license files beneath each native snapshot's `vendor/` directory. Some GGML backends carry additional per-file notices.
- **Optional FlyDSL/AITER-derived adapter:** its source identifies Apache-2.0; a copy is provided in `licenses/Apache-2.0.txt`. Generated GPU artifacts and the external compiler/runtime are not bundled.
- **Model weights:** not included or relicensed. Obtain model files and vision components from their respective upstream publishers and observe those terms. The Windows model reference remains in `windows/qwen27b/model.source.json`; the Linux operator supplies compatible local model artifacts.
- **Python dependencies:** requirements are listed in each service directory. Their own licenses apply.

No blanket new license is asserted over all original service code in this archive. Retained upstream licenses apply to the identified components; source availability does not by itself grant additional rights over other code or model weights. Public upstream copyright notices remain with their source files.
