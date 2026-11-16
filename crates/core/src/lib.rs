//! LLM inference on WebGPU, in the browser and natively.

pub mod engine;
pub mod model;
pub mod philox;

pub use engine::{Engine, MAX_CHUNK, MAX_CONTEXT, Sampling};
pub use model::{Config, Weights};
