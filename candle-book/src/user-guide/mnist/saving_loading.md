# Candle MNIST Tutorial

## Saving and Loading Models

After training a model, it is useful to save and subsequently load the model parameters. In Candle, this functionality is managed through the `VarMap` data structure, with parameters stored on disk using the [safetensors](https://huggingface.co/docs/safetensors/index) format.

### Saving Model Parameters

Let's modify our `training_loop` function to include functionality for saving weights:

```rust
{{#include code/mnist5.rs:51:103}}
```

```bash
$ cargo run --release

> 1 train loss:  2.40485 test acc:  0.11%
> 2 train loss:  2.34161 test acc:  0.14%
> 3 train loss:  2.28841 test acc:  0.17%
> 4 train loss:  2.24158 test acc:  0.19%
> 5 train loss:  2.19898 test acc:  0.23%
> 6 train loss:  2.15927 test acc:  0.26%
> 7 train loss:  2.12161 test acc:  0.29%
> 8 train loss:  2.08549 test acc:  0.32%
> 9 train loss:  2.05053 test acc:  0.35%
```

### Loading Model Parameters

Now that we have saved our model parameters, we can modify the code to load them. The primary change required is to make the `varmap` variable mutable:

```rust
{{#include code/mnist6.rs:51:106}}
```

```bash
$ cargo run --release

> 1 train loss:  2.01645 test acc:  0.38%
> 2 train loss:  1.98300 test acc:  0.41%
> 3 train loss:  1.95008 test acc:  0.44%
> 4 train loss:  1.91754 test acc:  0.47%
> 5 train loss:  1.88534 test acc:  0.50%
> 6 train loss:  1.85349 test acc:  0.53%
> 7 train loss:  1.82198 test acc:  0.56%
> 8 train loss:  1.79077 test acc:  0.59%
> 9 train loss:  1.75989 test acc:  0.61%
```

Note that loading the weights will fail if the specified file does not exist or is incompatible with the current model architecture. Implementing file existence checks and appropriate error handling is left to the user.