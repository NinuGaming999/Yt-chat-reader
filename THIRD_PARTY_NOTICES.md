# Third-party notices

NinuGaming Live Assistant bundles or uses the following third-party components:

- **Kokoro-82M v1.0 model and voice embeddings** — Apache License 2.0. Model files are from the official `thewh1teagle/kokoro-onnx` `model-files-v1.1` release.
- **kokoro-onnx 0.6.1** — MIT License.
- **ONNX Runtime DirectML** — MIT License; DirectML is the GPU execution provider on compatible Windows systems.
- **eSpeak NG via espeakng-loader / phonemizer** — see the licenses distributed with those packages and their upstream projects.
- **Flask, Requests, pywebview, NumPy and SoundDevice** — see their respective licenses in their installed distributions.
- **Microsoft Edge WebView2 Runtime** — Microsoft software distributed under Microsoft's applicable license terms; it is installed by its official bootstrapper when needed.

Official model files and SHA-256 checksums are listed in `download_models.ps1`. The application does not modify the upstream model weights.