# Tensor Storage

The Tensor [`Storage`](https://github.com/huggingface/candle/blob/main/candle-core/src/storage.rs#L10)
enum tells us where the data for a partiular tensor is stored.

```rust
{{#include ../../../../../candle-core/src/storage.rs:10:14}}
```

There are 3 backends you can use, either cpu, which stores tensor data
in ram, cuda, which can store tensor data on a cuda enabled GPU, or metal which
stores the data in a format to be used by metal on MacOS.

### Cpu Storage

The most intuive backend is the [`CpuStorage`](https://github.com/huggingface/candle/blob/main/candle-core/src/cpu_backend/mod.rs#L21) 
enum, where the data is stored in `Vec`s. 

```rust
{{#include ../../../../../candle-core/src/cpu_backend/mod.rs:21:30}}
```

### Cuda Storage

[CudaStorage](https://github.com/huggingface/candle/blob/main/candle-core/src/cuda_backend/mod.rs#L1132) uses [cudarc](https://github.com/coreylowman/cudarc) a rust wrapper
for the cuda toolkit. This is only active when compiled with cuda feature.

We can see the struct here

```rust
{{#include ../../../../../candle-core/src/cuda_backend/mod.rs:1132:1135}}
```

that contains a [CudaStorageSlice](https://github.com/huggingface/candle/blob/main/candle-core/src/cuda_backend/mod.rs#L66) which represents the actual memory

```rust
{{#include ../../../../../candle-core/src/cuda_backend/mod.rs:66:75}}
```

and a [CudaDevice](https://github.com/huggingface/candle/blob/main/candle-core/src/cuda_backend/device.rs#L34) which describes how to communicate with the device.

```rust
{{#include ../../../../../candle-core/src/cuda_backend/device.rs:34:42}}
```

### Metal Storage

[MetalStorage](https://github.com/huggingface/candle/blob/main/candle-core/src/metal_backend/mod.rs#L71) uses the deprecated [metal-rs](https://github.com/gfx-rs/metal-rs) a rust wrapper
for the unmaintained objc ecosystem of mac system bindings. 
This is only active when compiled with metal feature.

We can see the struct here

```rust
{{#include ../../../../../candle-core/src/metal_backend/mod.rs:71}}
{{#include ../../../../../candle-core/src/metal_backend/mod.rs:73}}
{{#include ../../../../../candle-core/src/metal_backend/mod.rs:75}}
{{#include ../../../../../candle-core/src/metal_backend/mod.rs:77}}
{{#include ../../../../../candle-core/src/metal_backend/mod.rs:79:80}}
```

This has a [MetalDevice](https://github.com/huggingface/candle/blob/main/candle-core/src/metal_backend/device.rs#L93) which describes how to communicate with the device.

```rust
{{#include ../../../../../candle-core/src/metal_backend/device.rs:93}}
{{#include ../../../../../candle-core/src/metal_backend/device.rs:96}}
{{#include ../../../../../candle-core/src/metal_backend/device.rs:99}}
{{#include ../../../../../candle-core/src/metal_backend/device.rs:101}}
{{#include ../../../../../candle-core/src/metal_backend/device.rs:116}}
{{#include ../../../../../candle-core/src/metal_backend/device.rs:120}}
{{#include ../../../../../candle-core/src/metal_backend/device.rs:122:123}}
```