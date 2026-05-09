# kokoro-tts

A lightweight, offline Rust inference library for [Kokoro TTS](https://github.com/hexgrad/kokoro) - an 82M-parameter open-weights text-to-speech model. Designed to be small, dependency-light, and easy to cross-compile to mobile.

> Kokoro is an open-weights TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.

---

## Features

- 100% offline - no network calls at runtime.
- Cross-platform - builds on macOS, Linux, Windows; cross-compiles to Android and iOS.
- Hardware acceleration - CoreML on macOS, CUDA on Linux/Windows, with automatic CPU fallback.
- Multiple model sizes - full-precision (`model.onnx`), 8-bit-quantized (`model_quantized.onnx`), and others.
- Multiple voices spanning English, Mandarin, Spanish, French, Japanese, Italian, Hindi, Brazilian Portuguese. Only battle tested with English.
- Streaming and one-shot synthesis modes.

---

## Quick start

### 1. Download the model and voices

The model and voice files are not bundled with this repository. Get them from the official ONNX release on Hugging Face:

> **<https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main>**

You need **two** sets of files:

| What            | Hugging Face folder                                | Goes into local folder |
| --------------- | -------------------------------------------------- | ---------------------- |
| ONNX model file | [`onnx/`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/onnx) | `models/`              |
| Voice packs     | [`voices/`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices) | `voices/`              |

Pick whichever variant of the model fits your hardware (any one is enough):

| File                   | Size  | Notes                                                      |
| ---------------------- | ----- | ---------------------------------------------------------- |
| `model.onnx`           | ~325 MB | Full precision (fp32). Best quality. Recommended for CoreML / CUDA. |
| `model_q8f16.onnx`     | ~160 MB | Mixed int8 / fp16. Smaller, fast on CPU.                   |
| `model_quantized.onnx` | ~92 MB  | int8. Smallest. Some quality loss.                         |

Each voice is a separate `<name>.bin` file (e.g. `af_heart.bin`, `af_alloy.bin`). Download as many or as few as you want - the library only loads what's in the folder.

You can grab them via `git lfs`:

```bash
git lfs install
git clone https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
mkdir -p models voices
cp Kokoro-82M-v1.0-ONNX/onnx/*.onnx       models/
cp Kokoro-82M-v1.0-ONNX/voices/*.bin      voices/
```

…or download individual files from the Hugging Face web UI and drop them into the matching folder.

After this step the layout should look like:

```
kokoro/
├── models/
│   ├── model.onnx                ← any one (or more) of these
│   ├── model_q8f16.onnx
│   └── model_quantized.onnx
└── voices/
    ├── af_alloy.bin
    ├── af_heart.bin
    ├── am_adam.bin
    └── …                         ← whichever voices you want
```

### 2. Install platform deps

- **macOS**: nothing extra required.
- **Linux**: `sudo apt install libasound2-dev` (only needed for the `voxudio` audio playback used by the examples).
- **Windows**: nothing extra required.

### 3. Run an example

```bash
cargo run --release --example synth_directly_v10
cargo run --release --example synth_stream
```

The first build downloads ONNX Runtime and compiles the bundled `cmudict` dictionary, so expect a couple of minutes. Subsequent builds are fast.

---

## Using the library in your own project

```bash
cargo add kokoro-tts
```

```rust
use kokoro_tts::{KokoroTts, Voice};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Point at the model file and the voices/ directory.
    // The directory will load every `<name>.bin` file inside it; you can
    // also pass a single `.bin` file to load just one voice.
    let tts = KokoroTts::new("models/model.onnx", "voices").await?;

    // Pick a voice by name. Anything that converts into `Voice` works
    // (`&str`, `String`, or `Voice::new(...)`).
    let (audio, took) = tts.synth("Hello, world!", Voice::new("af_heart")).await?;
    println!("Synth took {took:?}, produced {} samples at 24 kHz", audio.len());

    // The shorthand also works:
    let _ = tts.synth("Hello again!", "af_alloy").await?;

    // Speed control:
    let _ = tts.synth("Slower.", Voice::new("af_heart").with_speed(0.85)).await?;

    Ok(())
}
```

### Voice API at a glance

```rust
Voice::new("af_alloy")                  // default speed 1.0
Voice::new("af_alloy").with_speed(1.2)  // 20% faster
"af_alloy".into()                       // From<&str> implementation
```

Any function taking `Voice` actually accepts `impl Into<Voice>`, so you rarely need to construct one explicitly:

```rust
tts.synth("hi", "af_alloy").await?;
tts.stream("af_alloy");
```

### Streaming long text

```rust
use {futures::StreamExt, kokoro_tts::KokoroTts};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tts = KokoroTts::new("models/model_quantized.onnx", "voices").await?;
    let (mut sink, mut stream) = tts.stream("af_alloy");

    sink.synth("First sentence.").await?;
    sink.synth("Second sentence - synthesized while the first one plays.").await?;
    drop(sink);

    while let Some((audio, took)) = stream.next().await {
        println!("Got {} samples in {took:?}", audio.len());
    }

    Ok(())
}
```

---

## Voice naming convention

Voice names follow the upstream Kokoro convention `<lang><gender>_<name>`:

| Prefix | Language / accent              |
| ------ | ------------------------------ |
| `af`   | American English, female       |
| `am`   | American English, male         |
| `bf`   | British English, female        |
| `bm`   | British English, male          |
| `ef`   | Spanish, female                |
| `em`   | Spanish, male                  |
| `ff`   | French, female                 |
| `hf`   | Hindi, female                  |
| `hm`   | Hindi, male                    |
| `if`   | Italian, female                |
| `im`   | Italian, male                  |
| `jf`   | Japanese, female               |
| `jm`   | Japanese, male                 |
| `pf`   | Brazilian Portuguese, female   |
| `pm`   | Brazilian Portuguese, male     |
| `zf`   | Mandarin Chinese, female (v1.1)|
| `zm`   | Mandarin Chinese, male  (v1.1) |

Pick the voice file you want from the [Hugging Face `voices/` folder](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices) and reference it by its file-stem name (the filename without `.bin`).

---

## Execution providers

The library auto-selects the best available backend:

- **macOS** → CoreML (Neural Engine + GPU). Falls back to CPU if the model can't run on CoreML.
- **Linux / Windows** → CUDA if available, else CPU.

You can override or pin the provider with environment variables:

| Variable                              | Values                                                                      |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `KOKORO_ORT_PROVIDER`                 | `auto` (default), `cpu`, `coreml`, `cuda`                                   |
| `KOKORO_COREML_MODEL_FORMAT`          | `neuralnetwork` (default), `mlprogram`                                      |
| `KOKORO_COREML_COMPUTE_UNITS`         | `all` (default), `ane`, `gpu`, `cpu_only`                                   |
| `KOKORO_COREML_STATIC_INPUT_SHAPES`   | `0` (default) / `1`                                                         |

Setting `KOKORO_ORT_PROVIDER=coreml` or `=cuda` disables the automatic CPU fallback so failures surface explicitly.

---

## Troubleshooting

- **`VoiceNotFound("af_heart")`** - the file `voices/af_heart.bin` doesn't exist. Download it from the Hugging Face `voices/` folder.
- **`Io(... model.onnx ...)`** - the model file isn't where you said it was. Check the path you passed to `KokoroTts::new`.
- **`no .bin voice files found in voices`** - the directory is empty. Drop at least one `<name>.bin` file in there.
- **CoreML errors at startup** - set `KOKORO_ORT_PROVIDER=cpu` to force a known-good path. Quantized models don't run on CoreML; the library detects this and falls back to CPU automatically when `KOKORO_ORT_PROVIDER` is left at `auto`.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

The Kokoro model weights are also Apache-2.0; check the upstream Hugging Face repo for the canonical license text covering the weights themselves.
