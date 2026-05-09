use {
    kokoro_tts::{KokoroTts, Voice},
    voxudio::AudioPlayer,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tts = KokoroTts::new("model.onnx", "af_heart.bin").await?;
    let (audio, took) = tts
        .synth(
            "Hello, world! I'm a 25 year old software engineer with a passion background?",
            Voice::AfHeart(1.0),
        )
        .await?;
    println!("Synth took: {:?}", took);
    let mut player = AudioPlayer::new()?;
    player.play()?;
    player.write::<24000>(&audio, 1).await?;

    Ok(())
}
