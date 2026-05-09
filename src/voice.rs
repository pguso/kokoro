//! Voice selection for Kokoro TTS.
//!
//! Voices are referenced by their file-stem name — the same name as the
//! corresponding `.bin` file inside the `voices/` directory, without the
//! extension. For example the file `voices/af_heart.bin` is selected with
//! [`Voice::new("af_heart")`].
//!
//! ```no_run
//! use kokoro_en::{KokoroTts, Voice};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let tts = KokoroTts::new("models/model.onnx", "voices").await?;
//!
//! // Default speed (1.0).
//! let v = Voice::new("af_heart");
//!
//! // Custom speed.
//! let fast = Voice::new("af_heart").with_speed(1.2);
//!
//! // `&str` is accepted anywhere a `Voice` is expected via `Into<Voice>`.
//! let _ = tts.synth("Hello, world!", "af_heart").await?;
//! # Ok(()) }
//! ```

/// A voice referenced by its file-stem name plus an optional speech speed.
#[derive(Clone, Debug, PartialEq)]
pub struct Voice {
    pub(crate) name: String,
    pub(crate) speed: f32,
}

impl Voice {
    /// Build a voice referenced by its file-stem name (e.g. `"af_heart"`).
    /// Default speed is `1.0`.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            speed: 1.0,
        }
    }

    /// Override the speech speed (default `1.0`). Values >1.0 speak faster,
    /// values <1.0 speak slower. Negative values are clamped at the model
    /// boundary.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// The voice file-stem name as registered in the loaded voice pack.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The configured speech speed.
    pub fn speed(&self) -> f32 {
        self.speed
    }
}

impl From<&str> for Voice {
    fn from(s: &str) -> Self {
        Voice::new(s)
    }
}

impl From<String> for Voice {
    fn from(s: String) -> Self {
        Voice::new(s)
    }
}

impl From<&String> for Voice {
    fn from(s: &String) -> Self {
        Voice::new(s.clone())
    }
}
