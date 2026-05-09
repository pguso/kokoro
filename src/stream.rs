use {
    crate::{KokoroError, Voice},
    futures::{Sink, SinkExt, Stream},
    pin_project::pin_project,
    std::{
        pin::Pin,
        task::{Context, Poll},
        time::Duration,
    },
    tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
};

struct Request<S> {
    voice: Voice,
    text: S,
}

struct Response {
    data: Vec<f32>,
    took: Duration,
}

/// 语音合成流
///
/// 该结构体用于通过流式合成来处理更长的文本。它实现了`Stream` trait，可以用于异步迭代合成后的音频数据。
#[pin_project]
pub struct SynthStream {
    #[pin]
    rx: UnboundedReceiver<Response>,
}

impl Stream for SynthStream {
    type Item = (Vec<f32>, Duration);

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.project().rx)
            .poll_recv(cx)
            .map(|i| i.map(|Response { data, took }| (data, took)))
    }
}

/// 语音合成发送端
///
/// 该结构体用于发送语音合成请求。它实现了`Sink` trait，可以用于异步发送合成请求。
#[pin_project]
pub struct SynthSink<S> {
    tx: UnboundedSender<Request<S>>,
    voice: Voice,
}

impl<S> SynthSink<S> {
    /// Switch the active voice for subsequent synthesis requests.
    ///
    /// Accepts anything convertible into a [`Voice`], including `&str` and
    /// `String`, so a plain voice name works:
    ///
    /// ```no_run
    /// use kokoro_en::{KokoroTts, Voice};
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let Ok(tts) = KokoroTts::new("models/model.onnx", "voices").await else {
    ///         return;
    ///     };
    ///     let (mut sink, _) = tts.stream::<&str, _>("af_heart");
    ///     // Same voice but slower.
    ///     sink.set_voice(Voice::new("af_heart").with_speed(0.8));
    /// }
    /// ```
    pub fn set_voice<V: Into<Voice>>(&mut self, voice: V) {
        self.voice = voice.into()
    }

    /// Send a synthesis request for `text` using the currently configured voice.
    ///
    /// ```no_run
    /// use kokoro_en::KokoroTts;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let Ok(tts) = KokoroTts::new("models/model.onnx", "voices").await else {
    ///         return;
    ///     };
    ///     let (mut sink, _) = tts.stream("af_heart");
    ///     let _ = sink.synth("hello world.").await;
    /// }
    /// ```
    pub async fn synth(&mut self, text: S) -> Result<(), KokoroError> {
        self.send((self.voice.clone(), text)).await
    }
}

impl<S> Sink<(Voice, S)> for SynthSink<S> {
    type Error = KokoroError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, (voice, text): (Voice, S)) -> Result<(), Self::Error> {
        self.tx
            .send(Request { voice, text })
            .map_err(|e| KokoroError::Send(e.to_string()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

pub(super) fn start_synth_session<F, R, S>(
    voice: Voice,
    synth_request_callback: F,
) -> (SynthSink<S>, SynthStream)
where
    F: Fn(S, Voice) -> R + Send + 'static,
    R: Future<Output = Result<(Vec<f32>, Duration), KokoroError>> + Send,
    S: AsRef<str> + Send + 'static,
{
    let (tx, mut rx) = unbounded_channel::<Request<S>>();
    let (tx2, rx2) = unbounded_channel();
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            match synth_request_callback(req.text, req.voice).await {
                Ok((data, took)) => {
                    if let Err(e) = tx2.send(Response { data, took }) {
                        return Err(KokoroError::Send(e.to_string()));
                    }
                }
                Err(e) => {
                    // Keep the stream alive so later requests can still produce audio.
                    eprintln!("synth request failed: {}", e);
                }
            }
        }

        Ok::<_, KokoroError>(())
    });

    (SynthSink { tx, voice }, SynthStream { rx: rx2 })
}
