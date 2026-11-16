//! Checks the engine against the reference and measures its speed.
//!
//! peerinfer-bench check MODEL_DIR REF_DIR
//! peerinfer-bench speed MODEL_DIR [PROMPT_TOKENS] [GENERATED_TOKENS] [RUNS]

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use peerinfer_core::{Engine, Weights};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") if args.len() == 3 => {
            pollster::block_on(check(args[1].as_ref(), args[2].as_ref()))
        }
        Some("speed") if args.len() >= 2 => {
            let n = |i: usize, d: usize| {
                args.get(i)
                    .map_or(Ok(d), |s| s.parse())
                    .context("bad number")
            };
            pollster::block_on(speed(args[1].as_ref(), n(2, 128)?, n(3, 128)?, n(4, 5)?))
        }
        Some("profile") if args.len() >= 2 => {
            let n = args
                .get(2)
                .map_or(Ok(1), |s| s.parse())
                .context("bad number")?;
            pollster::block_on(profile(args[1].as_ref(), n))
        }
        _ => bail!(
            "usage: peerinfer-bench check MODEL_DIR REF_DIR | speed MODEL_DIR [PROMPT] [GEN] [RUNS]"
        ),
    }
}

async fn load(dir: &std::path::Path) -> Result<(Engine, f64)> {
    let start = Instant::now();
    let weights = Weights::read_dir(dir)?;
    let engine = Engine::new(&weights).await?;
    let secs = start.elapsed().as_secs_f64();
    let info = engine.adapter_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);
    Ok((engine, secs))
}

async fn check(model: &std::path::Path, reference: &std::path::Path) -> Result<()> {
    let tokens: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("tokens.json"))?)?;
    let ids = |k: &str| -> Vec<u32> {
        tokens[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let (prompt, generated) = (ids("prompt"), ids("generated"));
    let raw = std::fs::read(reference.join("logits.f32"))?;
    let want: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let (mut engine, _) = load(model).await?;
    let vocab = engine.config().vocab_size;
    ensure!(
        want.len() == generated.len() * vocab,
        "reference logits have the wrong size"
    );

    let mut worst = 0f32;
    let mut agree = 0;
    let mut next = engine.forward(&prompt).await?;
    for (step, &expected) in generated.iter().enumerate() {
        let got = engine.last_logits().await?;
        let w = &want[step * vocab..(step + 1) * vocab];
        let diff = got
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let range =
            w.iter().fold(f32::MIN, |m, &v| m.max(v)) - w.iter().fold(f32::MAX, |m, &v| m.min(v));
        worst = worst.max(diff / range);
        agree += usize::from(next == expected);
        println!(
            "step {step:2}: token {next:6} expected {expected:6}  max |diff| {diff:.4} ({:.4}% of range)",
            100.0 * diff / range
        );
        next = engine.forward(&[expected]).await?;
    }
    println!(
        "top-1 agreement {agree}/{}  worst diff {:.4}% of logit range",
        generated.len(),
        100.0 * worst
    );
    engine.reset();
    let first = engine.forward(&prompt).await?;
    let mut fast = vec![first];
    engine
        .generate(generated.len() - 1, |t| {
            fast.extend_from_slice(t);
            true
        })
        .await?;
    println!("generate matches the reference: {}", fast == generated);
    ensure!(fast == generated, "generate returned {fast:?}");
    ensure!(
        agree == generated.len(),
        "top-1 tokens differ from the reference"
    );
    Ok(())
}

async fn speed(
    model: &std::path::Path,
    prompt_len: usize,
    gen_len: usize,
    runs: usize,
) -> Result<()> {
    let (mut engine, load_s) = load(model).await?;
    println!("load {load_s:.2} s (read files, upload, compile)");
    // Fixed, ordinary token IDs; the content does not matter for speed.
    let prompt: Vec<u32> = (0..prompt_len as u32)
        .map(|i| 1000 + (i * 37) % 20000)
        .collect();

    let mut prefill = Vec::new();
    let mut decode = Vec::new();
    for run in 0..=runs {
        engine.reset();
        let t0 = Instant::now();
        engine.forward(&prompt).await?;
        let t1 = Instant::now();
        let mut produced = 0;
        engine
            .generate(gen_len, |t| {
                produced += t.len();
                true
            })
            .await?;
        ensure!(produced == gen_len, "generated {produced} tokens");
        let t2 = Instant::now();
        if run == 0 {
            continue; // warm-up
        }
        prefill.push(prompt_len as f64 / (t1 - t0).as_secs_f64());
        decode.push(gen_len as f64 / (t2 - t1).as_secs_f64());
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "prefill {:.0} tok/s ({prompt_len} tokens)  decode {:.1} tok/s ({gen_len} tokens)  median of {runs}",
        median(&mut prefill),
        median(&mut decode)
    );
    Ok(())
}

async fn profile(model: &std::path::Path, n: usize) -> Result<()> {
    let (mut engine, _) = load(model).await?;
    let prompt: Vec<u32> = (0..128u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let pass: Vec<u32> = prompt.iter().copied().cycle().take(n).collect();
    engine.forward(&prompt).await?;
    engine.profile(&pass).await?; // warm-up
    engine.reset();
    engine.forward(&prompt).await?;
    let mut rows = engine.profile(&pass).await?;
    if rows.is_empty() {
        bail!("this GPU does not support timestamp queries");
    }
    let total: f64 = rows.iter().map(|r| r.1).sum();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, ms) in rows {
        println!("{ms:7.3} ms {:5.1}%  {name}", 100.0 * ms / total);
    }
    println!("{total:7.3} ms GPU time for a pass over {n} tokens after 128 cached tokens");
    Ok(())
}
