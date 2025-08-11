# Candle MNIST Tutorial

## Modeling

Open `src/main.rs` in your project folder and insert the following code:

```rust
{{#include code/mnist1.rs}}
```

Execute the program with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

Since random inputs are provided, expect an incoherent output.

## Implementing a `Linear` Layer

To create a more sophisticated layer type, add a `bias` to the weight to construct the standard `Linear` layer.

Replace the entire content of `src/main.rs` with:

```rust
{{#include code/mnist2.rs}}
```

Execute again with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

## Utilizing `candle_nn`

Many classical layers (such as [Linear](https://github.com/huggingface/candle/blob/main/candle-nn/src/linear.rs)) are already implemented in [candle-nn](https://github.com/huggingface/candle/tree/main/candle-nn).

This `Linear` implementation follows PyTorch conventions for improved compatibility with existing models, utilizing the transpose of weights rather than direct weights.

Let's simplify our implementation. First, add `candle-nn` as a dependency:

```bash
$ cargo add --git https://github.com/huggingface/candle.git candle-nn
```

Now, replace the entire content of `src/main.rs` with:

```rust
{{#include code/mnist3.rs:1:35}}
```

Execute the final version:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```