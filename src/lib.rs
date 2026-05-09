mod error;
mod g2p;
mod stream;
mod synthesizer;
mod tokenizer;
mod transcription;
mod voice;

use {
    bincode::{config::standard, decode_from_slice},
    ndarray::Array,
    ort::{
        inputs,
        session::{Session, builder::SessionBuilder},
        value::{TensorElementType, TensorRef, ValueType},
    },
    std::{collections::HashMap, env, path::Path, sync::Arc, time::Duration},
    tokio::{fs::read, sync::Mutex},
};
pub use {error::*, g2p::*, stream::*, tokenizer::*, transcription::*, voice::*};

pub struct KokoroTts {
    model: Arc<Mutex<Session>>,
    voices: Arc<HashMap<String, Vec<Vec<Vec<f32>>>>>,
    /// `true` when the loaded model expects the v1.1 phoneme/token vocabulary
    /// (its `speed` input is `i32`); `false` for v1.0 models that use `f32`.
    is_v11: bool,
}

#[cfg(target_os = "macos")]
fn configure_execution_providers(builder: SessionBuilder) -> SessionBuilder {
    use ort::ep::{
        CoreML,
        coreml::{ComputeUnits, ModelFormat},
    };

    let requested = env::var("KOKORO_ORT_PROVIDER").unwrap_or_else(|_| "auto".to_owned());
    if requested.eq_ignore_ascii_case("cpu") {
        eprintln!("kokoro ort | using CPU provider (KOKORO_ORT_PROVIDER=cpu)");
        return builder;
    }
    if !(requested.eq_ignore_ascii_case("auto") || requested.eq_ignore_ascii_case("coreml")) {
        eprintln!(
            "kokoro ort | unknown KOKORO_ORT_PROVIDER={:?} on macOS, defaulting to auto",
            requested
        );
    }

    // Default to NeuralNetwork (works with the standard fp32 model on ANE).
    // Quantized variants like `model_q8f16.onnx` cannot run on any CoreML
    // configuration — `KokoroTts::new` probes the session and falls back to
    // the bare CPU EP automatically when CoreML cannot execute the graph.
    let model_format = match env::var("KOKORO_COREML_MODEL_FORMAT")
        .unwrap_or_else(|_| "neuralnetwork".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "mlprogram" | "ml_program" | "ml-program" => ModelFormat::MLProgram,
        "neuralnetwork" | "neural_network" | "neural-network" | "nn" => ModelFormat::NeuralNetwork,
        other => {
            eprintln!(
                "kokoro ort | unknown KOKORO_COREML_MODEL_FORMAT={:?}, using NeuralNetwork",
                other
            );
            ModelFormat::NeuralNetwork
        }
    };

    let compute_units = match env::var("KOKORO_COREML_COMPUTE_UNITS")
        .unwrap_or_else(|_| "all".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "all" => ComputeUnits::All,
        "ane" | "neural_engine" | "neural-engine" | "cpu_and_neural_engine"
        | "cpu-and-neural-engine" => ComputeUnits::CPUAndNeuralEngine,
        "gpu" | "cpu_and_gpu" | "cpu-and-gpu" => ComputeUnits::CPUAndGPU,
        "cpu_only" | "cpu-only" | "cpuonly" => ComputeUnits::CPUOnly,
        other => {
            eprintln!(
                "kokoro ort | unknown KOKORO_COREML_COMPUTE_UNITS={:?}, using ALL",
                other
            );
            ComputeUnits::All
        }
    };

    let static_input_shapes = env::var("KOKORO_COREML_STATIC_INPUT_SHAPES")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True" | "yes" | "on"))
        .unwrap_or(false);

    let coreml = CoreML::default()
        .with_model_format(model_format)
        .with_compute_units(compute_units)
        .with_static_input_shapes(static_input_shapes)
        .build();

    match builder.clone().with_execution_providers([coreml]) {
        Ok(builder) => {
            eprintln!(
                "kokoro ort | using CoreML execution provider (model_format={}, compute_units={}, static_input_shapes={})",
                match model_format {
                    ModelFormat::MLProgram => "MLProgram",
                    ModelFormat::NeuralNetwork => "NeuralNetwork",
                },
                match compute_units {
                    ComputeUnits::All => "ALL",
                    ComputeUnits::CPUAndNeuralEngine => "CPUAndNeuralEngine",
                    ComputeUnits::CPUAndGPU => "CPUAndGPU",
                    ComputeUnits::CPUOnly => "CPUOnly",
                },
                static_input_shapes
            );
            builder
        }
        Err(err) => {
            eprintln!(
                "kokoro ort | CoreML unavailable, falling back to CPU provider: {}",
                err
            );
            builder
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn configure_execution_providers(builder: SessionBuilder) -> SessionBuilder {
    use ort::execution_providers::CUDAExecutionProvider;

    let requested = env::var("KOKORO_ORT_PROVIDER").unwrap_or_else(|_| "auto".to_owned());
    if requested.eq_ignore_ascii_case("cpu") {
        eprintln!("kokoro ort | using CPU provider (KOKORO_ORT_PROVIDER=cpu)");
        return builder;
    }
    if !(requested.eq_ignore_ascii_case("auto") || requested.eq_ignore_ascii_case("cuda")) {
        eprintln!(
            "kokoro ort | unknown KOKORO_ORT_PROVIDER={:?}, defaulting to auto",
            requested
        );
    }

    match builder
        .clone()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
    {
        Ok(builder) => {
            eprintln!("kokoro ort | using CUDA execution provider");
            builder
        }
        Err(err) => {
            eprintln!(
                "kokoro ort | CUDA unavailable, falling back to CPU provider: {}",
                err
            );
            builder
        }
    }
}

/// True when the user is allowed an automatic CPU fallback.
///
/// Disabled by setting `KOKORO_ORT_PROVIDER=coreml` / `=cuda` (i.e. an explicit
/// hardware EP request). `auto` (the default) and unset both allow fallback.
fn allow_cpu_fallback() -> bool {
    let requested = env::var("KOKORO_ORT_PROVIDER").unwrap_or_else(|_| "auto".to_owned());
    !(requested.eq_ignore_ascii_case("coreml") || requested.eq_ignore_ascii_case("cuda"))
}

/// Whether the session's `speed` input is `i32` (v1.1 models) instead of `f32` (v1.0).
fn session_speed_is_i32(session: &Session) -> bool {
    session
        .inputs()
        .iter()
        .find(|input| input.name() == "speed")
        .and_then(|input| match input.dtype() {
            ValueType::Tensor { ty, .. } => Some(*ty == TensorElementType::Int32),
            _ => None,
        })
        .unwrap_or(false)
}

/// Run a single dummy inference against the session.
///
/// Used to detect EP-level failures (CoreML "Error in building plan",
/// "Error in dynamically resizing for sequence length", MIL unbounded-dim
/// errors, etc.) before the user sees them.
fn probe_run(session: &mut Session, seq_len: usize) -> Result<(), KokoroError> {
    let speed_i32 = session_speed_is_i32(session);

    let mut tokens = vec![0i64; seq_len];
    if seq_len >= 2 {
        for token in &mut tokens[1..seq_len - 1] {
            *token = 1;
        }
    }
    let input_ids = Array::from_shape_vec((1, seq_len), tokens)?;
    let style = Array::from_shape_vec((1, 256), vec![0f32; 256])?;

    if speed_i32 {
        let speed = Array::from_vec(vec![1i32]);
        session.run(inputs![
            "input_ids" => TensorRef::from_array_view(&input_ids)?,
            "style" => TensorRef::from_array_view(&style)?,
            "speed" => TensorRef::from_array_view(&speed)?,
        ])?;
    } else {
        let speed = Array::from_vec(vec![1.0f32]);
        session.run(inputs![
            "input_ids" => TensorRef::from_array_view(&input_ids)?,
            "style" => TensorRef::from_array_view(&style)?,
            "speed" => TensorRef::from_array_view(&speed)?,
        ])?;
    }
    Ok(())
}

/// Two-shot probe: verifies first-run compilation AND dynamic-shape resizing.
///
/// `Error in building plan` shows up on the first call.
/// `Error in dynamically resizing for sequence length` shows up only when the
/// second call uses a different sequence length than the first.
fn probe_session(session: &mut Session) -> Result<(), KokoroError> {
    probe_run(session, 5)?;
    probe_run(session, 12)?;
    Ok(())
}

/// Build a session with the configured EP, probe it, and on failure rebuild
/// with the bare CPU EP. Returns the working session.
fn build_and_probe_session(model_source: ModelSource<'_>) -> Result<Session, KokoroError> {
    let mut session = build_session(model_source.clone(), false)?;
    if let Err(err) = probe_session(&mut session) {
        if !allow_cpu_fallback() {
            return Err(err);
        }
        eprintln!(
            "kokoro ort | hardware EP cannot run this model ({}), rebuilding with CPU provider",
            err
        );
        session = build_session(model_source, true)?;
        // Verify the CPU fallback actually works so the caller doesn't get a
        // broken session in their hands.
        probe_session(&mut session)?;
        eprintln!("kokoro ort | now using CPU provider after fallback");
    }
    Ok(session)
}

#[derive(Clone)]
enum ModelSource<'a> {
    Path(&'a Path),
    Bytes(&'a [u8]),
}

fn build_session(source: ModelSource<'_>, force_cpu: bool) -> Result<Session, KokoroError> {
    let builder = Session::builder()?;
    let mut builder = if force_cpu {
        builder
    } else {
        configure_execution_providers(builder)
    };
    let session = match source {
        ModelSource::Path(path) => builder.commit_from_file(path)?,
        ModelSource::Bytes(bytes) => builder.commit_from_memory(bytes)?,
    };
    Ok(session)
}

const STYLE_SIZE: usize = 256;
const F32_SIZE: usize = std::mem::size_of::<f32>();
const FRAME_SIZE: usize = STYLE_SIZE * F32_SIZE;

/// 16 MiB cap on any single bincode-decoded blob. Voice files are typically
/// ~512 KiB; the cap exists to prevent `bincode` from panicking with
/// `capacity overflow` when an unrelated byte stream happens to decode as a
/// huge varint length. With a limit set, bincode returns `Err` instead of
/// panicking, letting the raw-f32 fallback path run.
const VOICE_DECODE_LIMIT: usize = 16 * 1024 * 1024;

fn voice_decode_config() -> impl bincode::config::Config {
    standard().with_limit::<VOICE_DECODE_LIMIT>()
}

/// Reinterpret a raw `[N, 256]` little-endian f32 buffer as the voice pack
/// format `[N][1][256]`. Returns `None` if the buffer length is not a multiple
/// of one style frame.
fn decode_raw_f32_voice_pack(bytes: &[u8]) -> Option<Vec<Vec<Vec<f32>>>> {
    if bytes.is_empty() || bytes.len() % FRAME_SIZE != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(FRAME_SIZE)
            .map(|frame| {
                let style = frame
                    .chunks_exact(F32_SIZE)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<_>>();
                vec![style]
            })
            .collect::<Vec<_>>(),
    )
}

/// Decode either a raw `[N, 256]` little-endian f32 voice file or a packed
/// bincode voice file (`Vec<Vec<Vec<f32>>>`).
///
/// Raw-f32 is tried first because it is what the per-voice `.bin` files
/// distributed with Kokoro use, and because attempting bincode on a raw f32
/// buffer can otherwise trigger a panic in `Vec::with_capacity` when the
/// first bytes parse as an enormous varint length.
fn decode_voice_pack(bytes: &[u8]) -> Result<Vec<Vec<Vec<f32>>>, KokoroError> {
    if let Some(pack) = decode_raw_f32_voice_pack(bytes) {
        return Ok(pack);
    }
    if let Ok((pack, _)) =
        decode_from_slice::<Vec<Vec<Vec<f32>>>, _>(bytes, voice_decode_config())
    {
        return Ok(pack);
    }
    Err(KokoroError::VoiceVersionInvalid(
        "Invalid single voice format".to_owned(),
    ))
}

/// Load voices from either a directory of `<name>.bin` files or a single file.
///
/// - Directory: every `*.bin` entry is loaded as one voice keyed by its file stem.
/// - File: tries packed `HashMap<String, _>` first, then falls back to a single
///   voice file (bincode or raw f32) keyed by its file stem.
async fn load_voices(path: &Path) -> Result<HashMap<String, Vec<Vec<Vec<f32>>>>, KokoroError> {
    let metadata = tokio::fs::metadata(path).await?;
    if metadata.is_dir() {
        let mut voices = HashMap::new();
        let mut entries = tokio::fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            if !entry_path.is_file() {
                continue;
            }
            if entry_path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let Some(stem) = entry_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned)
            else {
                continue;
            };
            let bytes = read(&entry_path).await?;
            match decode_voice_pack(&bytes) {
                Ok(pack) => {
                    voices.insert(stem, pack);
                }
                Err(err) => {
                    eprintln!(
                        "kokoro voices | skipping {}: {}",
                        entry_path.display(),
                        err
                    );
                }
            }
        }
        if voices.is_empty() {
            return Err(KokoroError::VoiceNotFound(format!(
                "no .bin voice files found in {}",
                path.display()
            )));
        }
        Ok(voices)
    } else {
        let bytes = read(path).await?;
        if let Ok((voices, _)) = decode_from_slice::<HashMap<String, Vec<Vec<Vec<f32>>>>, _>(
            &bytes,
            voice_decode_config(),
        ) {
            return Ok(voices);
        }
        let voice_pack = decode_voice_pack(&bytes)?;
        let voice_name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or(KokoroError::VoiceNotFound(
                "invalid voice file name".to_owned(),
            ))?
            .to_owned();
        Ok(HashMap::from([(voice_name, voice_pack)]))
    }
}

impl KokoroTts {
    fn get_pack<'a>(
        voices: &'a HashMap<String, Vec<Vec<Vec<f32>>>>,
        voice: &Voice,
    ) -> Result<&'a Vec<Vec<Vec<f32>>>, KokoroError> {
        let name = voice.name();
        if let Some(pack) = voices.get(name) {
            return Ok(pack);
        }
        // Single-voice compatibility mode: if only one voice pack is loaded
        // (e.g. loading `af_heart.bin` directly), use it regardless of requested voice.
        if voices.len() == 1 {
            if let Some(pack) = voices.values().next() {
                return Ok(pack);
            }
        }
        Err(KokoroError::VoiceNotFound(name.to_owned()))
    }

    /// Load a model and one or more voices.
    ///
    /// `voices_path` may point to:
    /// - A directory containing `<name>.bin` files (recommended): every voice
    ///   in the directory is loaded and selectable by its file-stem name.
    /// - A single combined voices file (legacy bincode `HashMap`).
    /// - A single voice `.bin` file (the file-stem becomes the voice name).
    pub async fn new<P: AsRef<Path>>(model_path: P, voices_path: P) -> Result<Self, KokoroError> {
        let voices = load_voices(voices_path.as_ref()).await?;
        let model = build_and_probe_session(ModelSource::Path(model_path.as_ref()))?;
        let is_v11 = session_speed_is_i32(&model);
        Ok(Self {
            model: Arc::new(model.into()),
            voices: Arc::new(voices),
            is_v11,
        })
    }

    pub async fn new_from_bytes<B>(model: B, voices: B) -> Result<Self, KokoroError>
    where
        B: AsRef<[u8]>,
    {
        let (voices, _) = decode_from_slice::<HashMap<String, Vec<Vec<Vec<f32>>>>, _>(
            voices.as_ref(),
            voice_decode_config(),
        )?;

        let model = build_and_probe_session(ModelSource::Bytes(model.as_ref()))?;
        let is_v11 = session_speed_is_i32(&model);
        Ok(Self {
            model: Arc::new(model.into()),
            voices: Arc::new(voices),
            is_v11,
        })
    }

    /// Synthesize `text` using `voice`. `voice` accepts anything that converts
    /// into a [`Voice`] — including `&str` and `String` — so you can write
    /// `tts.synth("Hello", "af_heart").await` directly.
    pub async fn synth<S, V>(&self, text: S, voice: V) -> Result<(Vec<f32>, Duration), KokoroError>
    where
        S: AsRef<str>,
        V: Into<Voice>,
    {
        let voice = voice.into();
        let pack = Self::get_pack(self.voices.as_ref(), &voice)?;
        synthesizer::synth(Arc::downgrade(&self.model), text, pack, &voice, self.is_v11).await
    }

    /// Open a streaming synthesis session for the given `voice`.
    pub fn stream<S, V>(&self, voice: V) -> (SynthSink<S>, SynthStream)
    where
        S: AsRef<str> + Send + 'static,
        V: Into<Voice>,
    {
        let voices = self.voices.clone();
        let model = self.model.clone();
        let is_v11 = self.is_v11;

        start_synth_session(voice.into(), move |text, voice| {
            let voices = voices.clone();
            let model = model.clone();
            async move {
                let pack = Self::get_pack(voices.as_ref(), &voice)?;
                synthesizer::synth(Arc::downgrade(&model), text, pack, &voice, is_v11).await
            }
        })
    }
}
