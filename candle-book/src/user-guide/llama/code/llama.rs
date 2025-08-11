use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let device = Device.CPU;
    let dtype = DType::F16;

    let api = Api::new()?;
    let model_id = "meta-llama/Llama-3.2-3B-Instruct".to_string();
    let revision = "main".to_string()

    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let tokenizer_filename = api.get("tokenizer.json")?;
    let config_filename = api.get("config.json")?;
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(false);

    let filenames = candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json");
    let mut cache = model::Cache::new(true, dtype, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let llama = Llama::load(vb, &config)?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id = config.eos_token_id;
    let prompt = "What is the capital of France?".to_string();
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    let mut logits_processor = {
        let sampling = Sampling::All { temperature: 0.8}
        LogitsProcessor::from_sampling(299792458, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..10000 {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = {
            let start_at = tokens.len().saturating_sub(128);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                1.1,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    Ok(())
}
