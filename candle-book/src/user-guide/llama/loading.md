# Candle Transformers Llama Tutorial

## Loading The Model From The Huggingface Hub

```rust
use candle_transformers::models::llama as model;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{Llama, LlamaConfig};
```