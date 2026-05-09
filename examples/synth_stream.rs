use {
    futures::StreamExt,
    kokoro_tts::{KokoroTts, Voice},
    voxudio::AudioPlayer,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tts = KokoroTts::new("model.onnx", "af_heart.bin").await?;
    let (mut sink, mut stream) = tts.stream(Voice::AfHeart(1.0));
    sink.synth("I'm a 25 year old AI product engineer with a passion for python and everything related to AI. I also have a strong background in computer science and mathematics.")
        .await?;

    sink.synth("I'm a 41 year old software engineer with a passion for python and everything related to AI. I also have a strong background in computer science and mathematics.")
        .await?;

    sink.synth("I'm a 50 year old software architect with a passion for python and everything related to AI. I also have a strong background in computer science and mathematics.")
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
