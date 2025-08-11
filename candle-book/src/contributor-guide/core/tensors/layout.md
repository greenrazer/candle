# Tensor Layout

We'll go into more detail in the storage section, but you can think of the data in a tensor as linear memory, basically a `Vec` of the element type.

The [`Layout`](https://github.com/huggingface/candle/blob/main/candle-core/src/layout.rs#L5) struct describes how that linear memory should be interpreted as a tensor.

```rust
{{#include ../../../../../candle-core/src/layout.rs:5:6}}
{{#include ../../../../../candle-core/src/layout.rs:8:10}}
```

It has 3 fields: `shape`, `stride`, and `offset`.

### Shape

This is the concept most people are familiar with when they think about a tensor.

[`Shape`](https://github.com/huggingface/candle/blob/main/candle-core/src/shape.rs#L6)

```rust
{{#include ../../../../../candle-core/src/shape.rs:6}}
```

It is just a list of numbers with length equal to the number of dimensions in a tensor. Each dimension has a corresponding size.

### Stride

The strides are initially computed from the shape. They describe how many jumps each index in a particular dimension moves through linear memory.

Strides are helpful cached values for performing:
- `raveling` (multi-dimensional index → linear/flat index)
- `unraveling` (linear/flat index → multi-dimensional index)

Focusing specifically on `raveling`:

For example, if you have a tensor with shape `[2, 3, 4]`, you will have a strides array `[12, 4, 1]`.

If you imagine the entire tensor in linear memory, it's basically a `Vec` that's 24 elements long. For every first index (e.g., `i` in `[i, j, k]`), we need to jump 12 elements. For every second index (e.g., `j` in `[i, j, k]`), we need to jump 4 elements, and so on.

Strides and offsets are necessary for optimized "view operations" like slices, skips, and permutes to avoid cloning data unnecessarily.

For example, if you wanted to permute our `[2, 3, 4]`-shaped tensor into a `[3, 2, 4]` tensor, normally you'd have to copy/move all the data into the new format. However, with strides, all we have to do is permute the strides along with the shape.

This works because the dot product between index and strides:

`linear_index = index · strides = index[0]*strides[0] + index[1]*strides[1] + index[2]*strides[2] + ...`

remains unchanged when we apply the same permutation to both vectors, since addition is commutative:

`linear_index = index · strides = index[1]*strides[1] + index[2]*strides[2] + ... + index[0]*strides[0] = index[0]*strides[0] + index[1]*strides[1] + index[2]*strides[2] + ...`

### Offset

Offset is initialized to zero at the beginning. It represents how far past the start of our data our actual data begins. This is useful because it makes slicing easy.

You can imagine slicing from the end `[:-4]`, where all you would have to do is adjust the shape.

But storing the entire offset as a `Vec<usize>` would be inefficient. If we could show that the `ravel_index` function is linear, then it would have the following property:

`ravel_index(nd_index_a + nd_index_b) = ravel_index(nd_index_a) + ravel_index(nd_index_b)`

This would mean instead of storing the entire `Vec<usize>` for the offset, we could just cache the linear offset into our 1D tensor.

This turns out to be true. Here's a quick proof outline for two dimensions:

```
ravel_index(a + b) = (a + b)[0]*strides[0] + (a + b)[1]*strides[1]
ravel_index(a + b) = (a[0] + b[0])*strides[0] + (a[1] + b[1])*strides[1]
ravel_index(a + b) = a[0]*strides[0] + b[0]*strides[0] + a[1]*strides[1] + b[1]*strides[1]
ravel_index(a + b) = a[0]*strides[0] + a[1]*strides[1] + b[0]*strides[0] + b[1]*strides[1]
ravel_index(a + b) = ravel_index(a) + ravel_index(b)
```

The proof for higher dimensions follows the same pattern and is left as an exercise for the reader.