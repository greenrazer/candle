# Candle Transformers Llama Tutorial

## Introduction

This tutorial provides an introduction to loading and use hub-models by loading
and running [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

## Getting Started

Before proceeding, please ensure that you have properly installed Candle by following the instructions in the [Installation](../installation.md) guide.

Also install `candle-transformers` and `candle-nn`:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-transformers candle_nn
```

And the [`hf-hub`](https://github.com/huggingface/hf-hub) crate:

```bash
cargo add hf-hub
```