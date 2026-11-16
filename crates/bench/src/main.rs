//! Checks the engine against the reference and measures its speed.
//!
//! peerinfer-bench check MODEL_DIR REF_DIR
//! peerinfer-bench speed MODEL_DIR [PROMPT_TOKENS] [GENERATED_TOKENS] [RUNS]

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use peerinfer_core::{Engine, Input, Output, Sampling, Weights, philox};

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
        Some("sample-check") if args.len() == 2 => {
            pollster::block_on(sample_check(args[1].as_ref()))
        }
        Some("speed-sample") if args.len() >= 2 => {
            let n = |i: usize, d: usize| {
                args.get(i)
                    .map_or(Ok(d), |s| s.parse())
                    .context("bad number")
            };
            pollster::block_on(speed_sample(
                args[1].as_ref(),
                n(2, 128)?,
                n(3, 128)?,
                n(4, 5)?,
            ))
        }
        Some("split-check") if args.len() == 4 => {
            pollster::block_on(split_check(args[1].as_ref(), args[2].as_ref(), &args[3]))
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

/// Checks the GPU sampler against an exact CPU computation.
async fn sample_check(model: &std::path::Path) -> Result<()> {
    let (mut engine, _) = load(model).await?;
    // Arbitrary tokens, so the next-token distribution is broad.
    let prompt: Vec<u32> = (0..64u32).map(|i| 1000 + (i * 37) % 20000).collect();
    engine.forward(&prompt).await?;
    for (t, top_p) in [(0.8, 0.9), (1.0, 1.0)] {
        println!("temperature {t}, top-p {top_p}:");
        sample_case(&mut engine, t, top_p).await?;
    }
    Ok(())
}

#[allow(non_snake_case)]
async fn sample_case(engine: &mut Engine, T: f32, TOP_P: f32) -> Result<()> {
    const DRAWS: u64 = 4000;
    let logits = engine.last_logits().await?;
    let pos = engine.position() as u32;

    // Exact nucleus in f64: sort by probability, keep the smallest prefix
    // with at least TOP_P of the mass.
    let m = logits.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
    let p: Vec<f64> = logits
        .iter()
        .map(|&l| ((l as f64 - m) / T as f64).exp())
        .collect();
    let total: f64 = p.iter().sum();
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let mut kept = vec![false; p.len()];
    let mut mass = 0.0;
    for &i in &order {
        kept[i] = true;
        mass += p[i];
        if mass >= TOP_P as f64 * total {
            break;
        }
    }
    let kept_mass: f64 = (0..p.len()).filter(|&i| kept[i]).map(|i| p[i]).sum();
    let n_kept = kept.iter().filter(|&&k| k).count();

    let mut counts = std::collections::HashMap::<u32, u64>::new();
    let (mut rng_ok, mut exact, mut outside) = (0, 0, 0);
    for seed in 0..DRAWS {
        engine.set_sampling(Some(Sampling {
            temperature: T,
            top_p: TOP_P,
            seed,
        }))?;
        let (tok, raw) = engine.resample().await?;
        let want_raw = philox::philox4x32_10([pos, 0, 0, 0], [seed as u32, (seed >> 32) as u32])[0];
        rng_ok += u64::from(raw == want_raw);
        outside += u64::from(!kept[tok as usize]);
        // The same draw on the CPU: cumulative mass in index order.
        let u = philox::uniform(seed, pos) as f64 * kept_mass;
        let mut acc = 0.0;
        let mut cpu = 0;
        for i in 0..p.len() {
            if kept[i] {
                acc += p[i];
                cpu = i as u32;
                if u < acc {
                    break;
                }
            }
        }
        exact += u64::from(cpu == tok);
        *counts.entry(tok).or_default() += 1;
    }
    let again = {
        engine.set_sampling(Some(Sampling {
            temperature: T,
            top_p: TOP_P,
            seed: 7,
        }))?;
        let a = engine.resample().await?;
        let b = engine.resample().await?;
        a == b
    };
    let tvd: f64 = 0.5
        * (0..p.len())
            .filter(|&i| kept[i] || counts.contains_key(&(i as u32)))
            .map(|i| {
                let want = if kept[i] { p[i] / kept_mass } else { 0.0 };
                let got = *counts.get(&(i as u32)).unwrap_or(&0) as f64 / DRAWS as f64;
                (want - got).abs()
            })
            .sum::<f64>();
    println!(
        "nucleus: {n_kept} tokens; {DRAWS} draws, {} distinct",
        counts.len()
    );
    println!("GPU random words equal to Philox4x32-10: {rng_ok}/{DRAWS}");
    println!("tokens outside the exact nucleus: {outside}");
    println!("same token as the CPU for the same random number: {exact}/{DRAWS}");
    println!(
        "total variation distance to the exact distribution: {tvd:.4} (sampling noise is about {:.4})",
        (n_kept as f64 / DRAWS as f64).sqrt() / 2.0
    );
    println!("same seed gives the same token: {again}");
    ensure!(
        rng_ok == DRAWS && outside == 0 && again,
        "sampler check failed"
    );
    Ok(())
}

async fn speed_sample(
    model: &std::path::Path,
    prompt_len: usize,
    gen_len: usize,
    runs: usize,
) -> Result<()> {
    let (mut engine, _) = load(model).await?;
    let prompt: Vec<u32> = (0..prompt_len as u32)
        .map(|i| 1000 + (i * 37) % 20000)
        .collect();
    let mut decode = Vec::new();
    for run in 0..=runs {
        engine.set_sampling(Some(Sampling {
            temperature: 0.7,
            top_p: 0.9,
            seed: run as u64,
        }))?;
        engine.reset();
        engine.forward(&prompt).await?;
        let t1 = Instant::now();
        engine.generate(gen_len, |_| true).await?;
        if run > 0 {
            decode.push(gen_len as f64 / t1.elapsed().as_secs_f64());
        }
    }
    decode.sort_by(f64::total_cmp);
    println!(
        "decode with temperature 0.7, top-p 0.9: {:.1} tok/s ({gen_len} tokens, median of {runs})",
        decode[decode.len() / 2]
    );
    Ok(())
}

/// Runs the model whole and split into stages at the layer numbers in
/// `cuts` (for example "15" or "10,20") and requires the same logits and
/// tokens at every step, greedy and with sampling.
async fn split_check(
    model: &std::path::Path,
    reference: &std::path::Path,
    cuts: &str,
) -> Result<()> {
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
    let all = Weights::read_dir(model)?;
    let total = all.config.num_hidden_layers;
    let mut bounds = vec![0];
    for c in cuts.split(',') {
        bounds.push(c.trim().parse::<usize>().context("bad cut")?);
    }
    bounds.push(total);
    ensure!(
        bounds.windows(2).all(|w| w[0] < w[1]),
        "cuts must increase within 1..{total}"
    );

    let mut whole = Engine::new(&all).await?;
    let mut stages = Vec::new();
    let all_bytes: usize = all.shards.iter().map(Vec::len).sum();
    for w in bounds.windows(2) {
        let weights = Weights::read_dir_stage(model, Some(w[0]..w[1]))?;
        let loaded = weights.shards.iter().filter(|s| !s.is_empty()).count();
        let bytes: usize = weights.shards.iter().map(Vec::len).sum();
        println!(
            "stage layers {:2}..{:2}: {loaded}/{} shards, {:.1} of {:.1} MB",
            w[0],
            w[1],
            weights.shards.len(),
            bytes as f64 / 1e6,
            all_bytes as f64 / 1e6
        );
        stages.push(Engine::new_stage(&weights, w[0]..w[1]).await?);
    }

    async fn chain(stages: &mut [Engine], tokens: &[u32]) -> Result<(u32, usize)> {
        let mut out = stages[0].run(Input::Tokens(tokens)).await?;
        let mut sent = 0;
        for s in &mut stages[1..] {
            let Output::Hidden(h) = out else {
                bail!("a middle stage returned a token")
            };
            sent += h.len() * 4;
            out = s.run(Input::Hidden(&h)).await?;
        }
        match out {
            Output::Token(t) => Ok((t, sent)),
            Output::Hidden(_) => bail!("the last stage returned hidden states"),
        }
    }

    // Greedy, teacher-forced with the reference tokens.
    let mut worst = 0f32;
    let mut input = prompt.clone();
    for (step, &expected) in generated.iter().enumerate() {
        let a = whole.forward(&input).await?;
        let (b, sent) = chain(&mut stages, &input).await?;
        let la = whole.last_logits().await?;
        let lb = stages.last().unwrap().last_logits().await?;
        let diff = la
            .iter()
            .zip(&lb)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        worst = worst.max(diff);
        ensure!(
            a == b,
            "step {step}: whole model picked {a}, stages picked {b}"
        );
        if step < 2 {
            println!(
                "step {step}: {} tokens in, {sent} bytes between stages",
                input.len()
            );
        }
        input = vec![expected];
    }
    println!(
        "greedy: same tokens for {} steps, largest logit difference {worst}",
        generated.len()
    );

    // Sampling: the last stage and the whole model use the same seed.
    let sampling = Some(Sampling {
        temperature: 0.8,
        top_p: 0.9,
        seed: 42,
    });
    whole.set_sampling(sampling)?;
    stages.last_mut().unwrap().set_sampling(sampling)?;
    whole.reset();
    for s in &mut stages {
        s.reset();
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let (mut ia, mut ib) = (prompt.clone(), prompt.clone());
    for _ in 0..24 {
        let ta = whole.forward(&ia).await?;
        let (tb, _) = chain(&mut stages, &ib).await?;
        a.push(ta);
        b.push(tb);
        ia = vec![ta];
        ib = vec![tb];
    }
    println!("sampling: same 24 tokens: {}", a == b);
    ensure!(a == b, "sampled tokens differ: {a:?} vs {b:?}");
    Ok(())
}
