//! LLM inference on WebGPU, in the browser and natively.

pub mod engine;
pub mod model;
pub mod philox;

pub use engine::{Engine, Input, MAX_CHUNK, MAX_CONTEXT, Output, Sampling};
pub use model::{Config, Weights};
