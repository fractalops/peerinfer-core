//! LLM inference on WebGPU, in the browser and natively.

pub mod engine;
pub mod model;

pub use engine::{Engine, MAX_CHUNK, MAX_CONTEXT};
pub use model::{Config, Weights};
