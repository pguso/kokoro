use {
    futures::StreamExt,
    kokoro_en::{KokoroTts, Voice},
    voxudio::AudioPlayer,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // `voices` is the directory of `<name>.bin` files; the voice name passed
    // below picks `voices/af_alloy.bin` automatically.
    let tts = KokoroTts::new("models/model_quantized.onnx", "voices").await?;

    let (mut sink, mut stream) = tts.stream(Voice::new("af_heart"));
    sink.synth("Build a Text-to-Image Generator (from Scratch) takes this philosophy and applies it to one of the most exciting areas in AI today: text-to-image generation.")
        .await?;
    drop(sink);

    let mut player = AudioPlayer::new()?;
    player.play()?;
    while let Some((audio, took)) = stream.next().await {
        player.write::<24000>(&audio, 1).await?;
        println!("Synth took: {:?}", took);
    }

    let _ = &tts;
    Ok(())
}
