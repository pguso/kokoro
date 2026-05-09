use {
    kokoro_tts::{KokoroTts, Voice},
    voxudio::AudioPlayer,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // `voices` can be either a single `.bin` file or a directory containing
    // `<name>.bin` files. Pointing at the `voices/` folder lets us select any
    // voice by its file-stem name at synth time.
    let tts = KokoroTts::new("models/model.onnx", "voices").await?;

    let (audio, took) = tts
        .synth(
            "Hello, world! I'm a 25 year old software engineer with a passion background?",
            Voice::new("af_heart"),
        )
        .await?;
    println!("Synth took: {:?}", took);

    let mut player = AudioPlayer::new()?;
    player.play()?;
    player.write::<24000>(&audio, 1).await?;

    Ok(())
}
