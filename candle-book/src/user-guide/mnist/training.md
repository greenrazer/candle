# Candle MNIST Tutorial

## Training Implementation

First, let's create a utility function `make_linear` that accepts a `VarBuilder` and returns an initialized linear layer. The `VarBuilder` constructs a `VarMap`, which is the data structure that stores our trainable parameters.

```rust
{{#include code/mnist4.rs:9:27}}
```

Next, let's implement a `new` method for our model class to accept a `VarBuilder` and initialize the model. We use `VarBuilder::pp` to "push prefix" so that the parameter names are organized hierarchically: the first layer weights as `first.weight` and `first.bias`, and the second layer weights as `second.weight` and `second.bias`.

```rust
{{#include code/mnist4.rs:29:46}}
```

Now, let's add the `candle-datasets` package to our project to access the MNIST dataset:

```bash
$ cargo add --git https://github.com/huggingface/candle.git candle-datasets
```

With the dataset available, we can implement our training loop:

```rust
{{#include code/mnist4.rs:48:100}}
```

Finally, let's implement our main function:

```rust
{{#include code/mnist4.rs:102:105}}
```

Let's execute the training process:

```bash
$ cargo run --release

> 1 train loss:  2.35449 test acc:  0.12%
> 2 train loss:  2.30760 test acc:  0.15%
> ...
```