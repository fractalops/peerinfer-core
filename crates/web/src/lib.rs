//! JavaScript bindings for the engine.

use peerinfer_core::{
    Input, Output, Weights,
    model::{Config, WeightIndex},
};
use wasm_bindgen::prelude::*;

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

#[wasm_bindgen]
pub struct Engine(peerinfer_core::Engine);

#[wasm_bindgen]
impl Engine {
    /// Uploads a model to the GPU. `shards` are the files named in
    /// `ndarray-cache.json`, in order.
    pub async fn load(
        chat_config: &str,
        ndarray_cache: &str,
        shards: Vec<js_sys::Uint8Array>,
    ) -> Result<Engine, JsError> {
        console_error_panic_hook::set_once();
        let weights = Weights {
            config: Config::from_mlc_chat_config(chat_config).map_err(js_err)?,
            index: WeightIndex::from_ndarray_cache(ndarray_cache).map_err(js_err)?,
            shards: shards.iter().map(|s| s.to_vec()).collect(),
        };
        let engine = peerinfer_core::Engine::new(&weights)
            .await
            .map_err(js_err)?;
        Ok(Engine(engine))
    }

    /// Uploads only `start..end` of the model's layers. `shards` holds the
    /// files named in `ndarray-cache.json`, in order; files the stage does not
    /// need (see `shardsFor`) may be empty.
    #[wasm_bindgen(js_name = loadStage)]
    pub async fn load_stage(
        chat_config: &str,
        ndarray_cache: &str,
        shards: Vec<js_sys::Uint8Array>,
        start: usize,
        end: usize,
    ) -> Result<Engine, JsError> {
        console_error_panic_hook::set_once();
        let weights = Weights {
            config: Config::from_mlc_chat_config(chat_config).map_err(js_err)?,
            index: WeightIndex::from_ndarray_cache(ndarray_cache).map_err(js_err)?,
            shards: shards.iter().map(|s| s.to_vec()).collect(),
        };
        let engine = peerinfer_core::Engine::new_stage(&weights, start..end)
            .await
            .map_err(js_err)?;
        Ok(Engine(engine))
    }

    /// Indexes of the shard files that layers `start..end` need.
    #[wasm_bindgen(js_name = shardsFor)]
    pub fn shards_for(
        chat_config: &str,
        ndarray_cache: &str,
        start: usize,
        end: usize,
    ) -> Result<Vec<u32>, JsError> {
        let config = Config::from_mlc_chat_config(chat_config).map_err(js_err)?;
        let index = WeightIndex::from_ndarray_cache(ndarray_cache).map_err(js_err)?;
        Ok(index
            .shards_for(&(start..end), config.num_hidden_layers)
            .into_iter()
            .map(|i| i as u32)
            .collect())
    }

    #[wasm_bindgen(getter, js_name = isFirst)]
    pub fn is_first(&self) -> bool {
        self.0.is_first()
    }

    #[wasm_bindgen(getter, js_name = isLast)]
    pub fn is_last(&self) -> bool {
        self.0.is_last()
    }

    #[wasm_bindgen(getter, js_name = hiddenSize)]
    pub fn hidden_size(&self) -> usize {
        self.0.config().hidden_size
    }

    /// Runs the first stage on tokens. Returns the next token (a number) on
    /// the last stage, otherwise the hidden states as a Float32Array.
    #[wasm_bindgen(js_name = runTokens)]
    pub async fn run_tokens(&mut self, tokens: Vec<u32>) -> Result<JsValue, JsError> {
        let out = self.0.run(Input::Tokens(&tokens)).await.map_err(js_err)?;
        Ok(output(out))
    }

    /// Runs a later stage on the hidden states from the stage before it.
    /// Returns the next token on the last stage, otherwise hidden states.
    #[wasm_bindgen(js_name = runHidden)]
    pub async fn run_hidden(&mut self, hidden: Vec<f32>) -> Result<JsValue, JsError> {
        let out = self.0.run(Input::Hidden(&hidden)).await.map_err(js_err)?;
        Ok(output(out))
    }

    /// Appends tokens to the context and returns the most likely next token.
    pub async fn forward(&mut self, tokens: Vec<u32>) -> Result<u32, JsError> {
        self.0.forward(&tokens).await.map_err(js_err)
    }

    /// Generates up to `max` tokens after the token returned by the last
    /// `forward` call.
    pub async fn generate(&mut self, max: usize) -> Result<Vec<u32>, JsError> {
        let mut out = Vec::with_capacity(max);
        self.0
            .generate(max, |t| {
                out.extend_from_slice(t);
                true
            })
            .await
            .map_err(js_err)?;
        Ok(out)
    }

    /// Generates up to `max` tokens after the token returned by the last
    /// `forward` call. Calls `on_tokens` with each batch as a Uint32Array.
    /// Generation stops when it returns false.
    #[wasm_bindgen(js_name = generateStream)]
    pub async fn generate_stream(
        &mut self,
        max: usize,
        on_tokens: js_sys::Function,
    ) -> Result<(), JsError> {
        let mut error = None;
        self.0
            .generate(max, |t| {
                let batch = js_sys::Uint32Array::from(t);
                match on_tokens.call1(&JsValue::NULL, &batch) {
                    Ok(v) => v.as_bool().unwrap_or(false),
                    Err(e) => {
                        error = Some(e);
                        false
                    }
                }
            })
            .await
            .map_err(js_err)?;
        match error {
            Some(e) => Err(JsError::new(&format!("{e:?}"))),
            None => Ok(()),
        }
    }

    /// Sets random sampling. A temperature of 0 selects greedy decoding.
    #[wasm_bindgen(js_name = setSampling)]
    pub fn set_sampling(&mut self, temperature: f32, top_p: f32, seed: u32) -> Result<(), JsError> {
        let s = (temperature > 0.0).then_some(peerinfer_core::Sampling {
            temperature,
            top_p,
            seed: u64::from(seed),
        });
        self.0.set_sampling(s).map_err(js_err)
    }

    pub fn reset(&mut self) {
        self.0.reset();
    }

    pub fn adapter(&self) -> String {
        let i = self.0.adapter_info();
        format!("{} {}", i.vendor, i.name)
    }
}

fn output(out: Output) -> JsValue {
    match out {
        Output::Token(t) => JsValue::from(t),
        Output::Hidden(h) => js_sys::Float32Array::from(h.as_slice()).into(),
    }
}
