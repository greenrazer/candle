# Tensors

In candle a [`Tensor`](https://github.com/huggingface/candle/blob/main/candle-core/src/tensor.rs#L68) is a `Arc` wrapped `Tensor_`.

```rust
{{#include ../../../../../candle-core/src/tensor.rs:68}}
```

And a [`Tensor_`](https://github.com/huggingface/candle/blob/main/candle-core/src/tensor.rs#L23) is a struct that contains

```rust
{{#include ../../../../../candle-core/src/tensor.rs:23:24}}
{{#include ../../../../../candle-core/src/tensor.rs:37:43}}
```

This is a lot, but at it's core the tensor implementation really boils down to 2
main fields: `layout` and `storage`.