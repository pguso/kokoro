mod error;
mod g2p;
mod stream;
mod synthesizer;
mod tokenizer;
mod transcription;
mod voice;

use {
    bincode::{config::standard, decode_from_slice},
    ort::session::{Session, builder::SessionBuilder},
    std::{collections::HashMap, env, path::Path, sync::Arc, time::Duration},
    tokio::{fs::read, sync::Mutex},
};
pub use {error::*, g2p::*, stream::*, tokenizer::*, transcription::*, voice::*};

pub struct KokoroTts {
    model: Arc<Mutex<Session>>,
    voices: Arc<HashMap<String, Vec<Vec<Vec<f32>>>>>,
}

#[cfg(target_os = "macos")]
fn configure_execution_providers(builder: SessionBuilder) -> SessionBuilder {
    use ort::execution_providers::CoreMLExecutionProvider;

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

    match builder
        .clone()
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
    {
        Ok(builder) => {
            eprintln!("kokoro ort | using CoreML execution provider");
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

impl KokoroTts {
    fn get_pack<'a>(
        voices: &'a HashMap<String, Vec<Vec<Vec<f32>>>>,
        voice: Voice,
    ) -> Result<&'a Vec<Vec<Vec<f32>>>, KokoroError> {
        let name = voice.get_name();
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

    pub async fn new<P: AsRef<Path>>(model_path: P, voices_path: P) -> Result<Self, KokoroError> {
        let voices_path = voices_path.as_ref();
        let voices = read(voices_path).await?;
        let voices = match decode_from_slice::<HashMap<String, Vec<Vec<Vec<f32>>>>, _>(
            &voices,
            standard(),
        ) {
            Ok((voices, _)) => voices,
            Err(_) => {
                // Compatibility fallback: allow loading a single voice pack file (e.g. `af_heart.bin`)
                // by keying it with the file stem (`af_heart`), so existing `Voice` names still resolve.
                let voice_pack =
                    match decode_from_slice::<Vec<Vec<Vec<f32>>>, _>(&voices, standard()) {
                        Ok((voice_pack, _)) => voice_pack,
                        Err(_) => {
                            // Also support raw f32 single-voice files laid out as [N, 256].
                            // Convert them to the library format [N][1][256].
                            const STYLE_SIZE: usize = 256;
                            const F32_SIZE: usize = std::mem::size_of::<f32>();
                            let frame_size = STYLE_SIZE * F32_SIZE;
                            if voices.len() % frame_size != 0 {
                                return Err(KokoroError::VoiceVersionInvalid(
                                    "Invalid single voice format".to_owned(),
                                ));
                            }
                            voices
                                .chunks_exact(frame_size)
                                .map(|frame| {
                                    let style = frame
                                        .chunks_exact(F32_SIZE)
                                        .map(|chunk| {
                                            f32::from_le_bytes([
                                                chunk[0], chunk[1], chunk[2], chunk[3],
                                            ])
                                        })
                                        .collect::<Vec<_>>();
                                    vec![style]
                                })
                                .collect::<Vec<_>>()
                        }
                    };
                let voice_name = voices_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .ok_or(KokoroError::VoiceNotFound(
                        "invalid voice file name".to_owned(),
                    ))?
                    .to_owned();
                HashMap::from([(voice_name, voice_pack)])
            }
        };

        let builder = Session::builder()?;
        let model = configure_execution_providers(builder).commit_from_file(model_path)?;
        Ok(Self {
            model: Arc::new(model.into()),
            voices: Arc::new(voices),
        })
    }

    pub async fn new_from_bytes<B>(model: B, voices: B) -> Result<Self, KokoroError>
    where
        B: AsRef<[u8]>,
    {
        let (voices, _) = decode_from_slice(voices.as_ref(), standard())?;

        let builder = Session::builder()?;
        let model = configure_execution_providers(builder).commit_from_memory(model.as_ref())?;
        Ok(Self {
            model: Arc::new(model.into()),
            voices: Arc::new(voices),
        })
    }

    pub async fn synth<S>(&self, text: S, voice: Voice) -> Result<(Vec<f32>, Duration), KokoroError>
    where
        S: AsRef<str>,
    {
        let pack = Self::get_pack(self.voices.as_ref(), voice)?;
        synthesizer::synth(Arc::downgrade(&self.model), text, pack, voice).await
    }

    pub fn stream<S>(&self, voice: Voice) -> (SynthSink<S>, SynthStream)
    where
        S: AsRef<str> + Send + 'static,
    {
        let voices = self.voices.clone();
        let model = self.model.clone();

        start_synth_session(voice, move |text, voice| {
            let voices = voices.clone();
            let model = model.clone();
            async move {
                let pack = Self::get_pack(voices.as_ref(), voice)?;
                synthesizer::synth(Arc::downgrade(&model), text, pack, voice).await
            }
        })
    }
}
