//! JavaScript bindings for the engine.

use peerinfer_core::{
    Weights,
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

    /// Appends tokens to the context and returns the most likely next token.
    pub async fn forward(&mut self, tokens: Vec<u32>) -> Result<u32, JsError> {
        self.0.forward(&tokens).await.map_err(js_err)
    }

    pub fn reset(&mut self) {
        self.0.reset();
    }

    pub fn adapter(&self) -> String {
        let i = self.0.adapter_info();
        format!("{} {}", i.vendor, i.name)
    }
}
