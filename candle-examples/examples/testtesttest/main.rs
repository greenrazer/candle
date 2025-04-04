#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::io::Write;
use std::time::Instant;
use tokenizers::Tokenizer;

use candle::quantized::gguf_file;
use candle::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_examples::token_output_stream::TokenOutputStream;
// Import both model types
use candle_transformers::models::quantized_llama as llama_model;
use candle_transformers::models::quantized_qwen2 as qwen_model;

const DEFAULT_PROMPT: &str = "Explain the concept of machine learning in simple terms.";
const SAMPLE_LEN: usize = 100;

// Use an enum to distinguish between model types
enum ModelType {
    Llama,
    Qwen,
}

struct ModelConfig {
    repo: &'static str,
    filename: &'static str,
    tokenizer_repo: &'static str,
    model_type: ModelType,
    is_chat_template: bool,
    eos_token: &'static str,
    gqa: usize,
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

fn get_tokenizer(repo: &str) -> anyhow::Result<Tokenizer> {
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(repo.to_string());
    let tokenizer_path = api.get("tokenizer.json")?;
    Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
}

fn get_model_path(repo: &str, filename: &str) -> anyhow::Result<std::path::PathBuf> {
    let api = hf_hub::api::sync::Api::new()?;
    let path = api.repo(hf_hub::Repo::with_revision(
        repo.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ))
    .get(filename)?;
    Ok(path)
}

// Trait to provide a common interface for both model types
trait ModelInterface {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle::Result<Tensor>;
}

// Implement the trait for LLaMA model
impl ModelInterface for llama_model::ModelWeights {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle::Result<Tensor> {
        self.forward(input, offset)
    }
}

// Implement the trait for Qwen model
impl ModelInterface for qwen_model::ModelWeights {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle::Result<Tensor> {
        self.forward(input, offset)
    }
}

// Enum to hold the actual model
enum Model {
    Llama(llama_model::ModelWeights),
    Qwen(qwen_model::ModelWeights),
}

// Implement the ModelInterface for the Model enum
impl ModelInterface for Model {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle::Result<Tensor> {
        match self {
            Model::Llama(model) => model.forward(input, offset),
            Model::Qwen(model) => model.forward(input, offset),
        }
    }
}

fn run_model(config: &ModelConfig) -> anyhow::Result<()> {
    // Add more robust error handling throughout this function
    let result = std::panic::catch_unwind(|| -> anyhow::Result<()> {
    println!("\n==================================================");
    println!("Running model: {}/{}", config.repo, config.filename);
    println!("==================================================\n");

    // Get the model path and tokenizer
    let model_path = get_model_path(config.repo, config.filename)?;
    let tokenizer = get_tokenizer(config.tokenizer_repo)?;
    let mut tos = TokenOutputStream::new(tokenizer);
    
    // Open and load the model
    let mut file = std::fs::File::open(&model_path)?;
    let start = Instant::now();
    let device = candle_examples::device(false)?; // Always use GPU if available
    
    // Read the GGUF model
    let gguf_model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
    
    // Calculate and display model size
    let mut total_size_in_bytes = 0;
    for (_, tensor) in gguf_model.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    println!(
        "Loaded {:?} tensors ({}) in {:.2}s",
        gguf_model.tensor_infos.len(),
        &format_size(total_size_in_bytes),
        start.elapsed().as_secs_f32(),
    );
    
    // Create the appropriate model based on the configuration
    let mut model = match config.model_type {
        ModelType::Llama => {
            println!("Building LLaMA model");
            Model::Llama(llama_model::ModelWeights::from_gguf(gguf_model, &mut file, &device)?)
        },
        ModelType::Qwen => {
            println!("Building Qwen model");
            Model::Qwen(qwen_model::ModelWeights::from_gguf(gguf_model, &mut file, &device)?)
        },
    };
    
    println!("Model built");
    
    // Format the prompt according to the model type
    let prompt_str = match (&config.model_type, config.is_chat_template) {
        (ModelType::Qwen, _) => {
            format!("<｜User｜>{}<｜Assistant｜>", DEFAULT_PROMPT)
        },
        (ModelType::Llama, true) => {
            format!("[INST] {} [/INST]", DEFAULT_PROMPT)
        },
        _ => DEFAULT_PROMPT.to_string(),
    };
    
    print!("Prompt: {}", &prompt_str);
    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    
    let prompt_tokens = tokens.get_ids().to_vec();
    let to_sample = SAMPLE_LEN.saturating_sub(1);
    
    // Ensure prompt fits within the context window
    let max_seq_len = match config.model_type {
        ModelType::Llama => llama_model::MAX_SEQ_LEN,
        ModelType::Qwen => 32768, // Typical Qwen context window size
    };
    
    let prompt_tokens = if prompt_tokens.len() + to_sample > max_seq_len - 10 {
        let to_remove = prompt_tokens.len() + to_sample + 10 - max_seq_len;
        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
    } else {
        prompt_tokens
    };
    
    // Process prompt and get first token
    let start_prompt_processing = Instant::now();
    let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    
    let mut logits_processor = LogitsProcessor::from_sampling(
        299792458, // fixed seed
        Sampling::TopP { p: 0.9, temperature: 0.8 }, // fixed parameters
    );
    
    let mut next_token = logits_processor.sample(&logits)?;
    let prompt_dt = start_prompt_processing.elapsed();
    
    let mut all_tokens = vec![next_token];
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }
    
    // Get the EOS token (safely)
    let eos_token = match tos.tokenizer().get_vocab(true).get(config.eos_token) {
        Some(token_id) => *token_id,
        None => {
            println!("Warning: Could not find EOS token '{}', using default fallback", config.eos_token);
            // Fallback - use a common token ID that's unlikely to appear in regular text
            100 // Arbitrary fallback value
        }
    };
    
    // Generate tokens
    let start_post_prompt = Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        
        // Apply repeat penalty
        let repeat_penalty = 1.1;
        let repeat_last_n = 64;
        let start_at = all_tokens.len().saturating_sub(repeat_last_n);
        let logits = candle_transformers::utils::apply_repeat_penalty(
            &logits,
            repeat_penalty,
            &all_tokens[start_at..],
        )?;
        
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        };
    }
    
    if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    
    let dt = start_post_prompt.elapsed();
    println!(
        "{sampled:4} tokens generated: {:.2} token/s, {:.2}",
        sampled as f64 / dt.as_secs_f64(), dt.as_secs_f64(),
    );
    
        Ok(())
    });
    
    match result {
        Ok(inner_result) => inner_result,
        Err(e) => {
            println!("Model execution panicked: {:?}", e);
            Ok(()) // Continue to the next model instead of stopping everything
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Enable CUDA optimizations
    candle::cuda::set_gemm_reduced_precision_f16(true);
    candle::cuda::set_gemm_reduced_precision_bf16(true);
    
    println!(
        "System info: avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    
    // Define all the models to run
    let models = [
        // ModelConfig {
        //     repo: "TheBloke/Mistral-7B-v0.1-GGUF",
        //     filename: "mistral-7b-v0.1.Q8_0.gguf",
        //     tokenizer_repo: "mistralai/Mistral-7B-v0.1",
        //     model_type: ModelType::Llama,
        //     is_chat_template: true,
        //     eos_token: "</s>",
        //     gqa: 8,
        // },
        // ModelConfig {
        //     repo: "TheBloke/Mistral-7B-v0.1-GGUF",
        //     filename: "mistral-7b-v0.1.Q4_0.gguf",
        //     tokenizer_repo: "mistralai/Mistral-7B-v0.1",
        //     model_type: ModelType::Llama,
        //     is_chat_template: true,
        //     eos_token: "</s>",
        //     gqa: 8,
        // },
        // ModelConfig {
        //     repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        //     filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        //     tokenizer_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        //     model_type: ModelType::Llama,
        //     is_chat_template: false,
        //     eos_token: "</s>",
        //     gqa: 1,
        // },
        // ModelConfig {
        //     repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        //     filename: "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        //     tokenizer_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        //     model_type: ModelType::Llama,
        //     is_chat_template: false,
        //     eos_token: "</s>",
        //     gqa: 1,
        // },
        // ModelConfig {
        //     repo: "Jaward/phi-3-mini-4k-instruct.Q4_0.gguf",
        //     filename: "phi-3-mini-4k-instruct.Q4_0.gguf",
        //     tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct",
        //     model_type: ModelType::Llama,
        //     is_chat_template: false,
        //     eos_token: "</s>",
        //     gqa: 1,
        // },
        // ModelConfig {
        //     repo: "mradermacher/SmolLM-135M-i1-GGUF",
        //     filename: "SmolLM-135M.i1-Q4_0.gguf",
        //     tokenizer_repo: "HuggingFaceTB/SmolLM-135M",
        //     model_type: ModelType::Llama,
        //     is_chat_template: false,
        //     eos_token: "</s>",
        //     gqa: 1,
        // },
        // ModelConfig {
        //     repo: "mradermacher/SmolLM-360M-i1-GGUF",
        //     filename: "SmolLM-360M.i1-Q4_0.gguf",
        //     tokenizer_repo: "HuggingFaceTB/SmolLM-360M",
        //     model_type: ModelType::Llama,
        //     is_chat_template: false,
        //     eos_token: "</s>",
        //     gqa: 1,
        // },
        // ModelConfig {
        //     repo: "mradermacher/SmolLM-360M-i1-GGUF",
        //     filename: "SmolLM-360M.i1-Q4_1.gguf",
        //     tokenizer_repo: "HuggingFaceTB/SmolLM-360M",
        //     model_type: ModelType::Llama,
        //     is_chat_template: false,
        //     eos_token: "</s>",
        //     gqa: 1,
        // },
        // ModelConfig {
        //     repo: "mradermacher/SmolLM-360M-i1-GGUF",
        //     filename: "SmolLM-360M.i1-Q6_K.gguf",
        //     tokenizer_repo: "HuggingFaceTB/SmolLM-360M",
        //     model_type: ModelType::Llama,
        //     is_chat_template: false,
        //     eos_token: "</s>",
        //     gqa: 1,
        // },
        // ModelConfig {
        //     repo: "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        //     filename: "DeepSeek-R1-Distill-Qwen-32B-Q4_0.gguf",
        //     tokenizer_repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        //     model_type: ModelType::Qwen,
        //     is_chat_template: false,
        //     eos_token: "<｜end▁of▁sentence｜>",
        //     gqa: 1,
        // },
    ];
    
    // Run each model
    for model_config in &models {
        match run_model(model_config) {
            Ok(_) => (),
            Err(e) => println!("Error running model {}/{}: {}", 
                model_config.repo, model_config.filename, e),
        }
    }
    
    println!("\n==================================================");
    println!("Benchmark complete for all models");
    println!("==================================================");
    
    Ok(())
}