# 🌹 Rosalia

A high-performance RWKV model inference engine implemented in Rust with Vulkan compute shaders.

## Overview

Rosalia is a Rust implementation of the **Rosa** mechanism from [RWKV-LM](https://github.com/BlinkDL/RWV-LM/tree/main/RWKV-v8), designed to provide efficient inference for RWKV language models using Vulkan GPU compute.

### What is RWKV?

RWKV (Receptance Weighted Key Value) is an innovative neural network architecture that combines the best of RNNs and Transformers. It offers:
- **Linear inference complexity** - unlike Transformers' quadratic attention
- **RNN-like efficient inference** - constant memory usage regardless of sequence length
- **Transformer-like training parallelism** - efficient training on modern hardware

The **Rosa** mechanism is a key component of RWKV-v8 that optimizes the model's recurrent computations.

## Features

- 🦀 **Pure Rust** - Memory-safe and high-performance implementation
- 🎮 **Vulkan Compute** - Cross-platform GPU acceleration
- ⚡ **Cooperative Matrix** - Utilizes Vulkan's cooperative matrix extension for optimized matrix operations
- 🔢 **Flexible Precision** - Supports FP16 and FP32 computation
- 📦 **Safetensors Support** - Load model weights in safetensors format
- 🧮 **Custom Layout System** - Efficient tensor layout management for GPU computation

## Architecture

```
Rosalia
├── app.rs        # Vulkan application core (device, memory, tensors, kernels)
├── layout.rs     # Tensor layout system with tiling and transformations
├── num.rs        # Data types (F16, F32, I8, I32, U8, U32)
├── asset.rs      # Embedded shader resources
└── assets/
    └── shaders/
        ├── src/
        │   └── gemm.comp    # GEMM compute shader source
        ├── spv/
        │   ├── gemm_f16_f16.spv  # FP16 input/output
        │   └── gemm_f16_f32.spv  # FP16 input, FP32 output
        └── compile.bat      # Shader compilation script
```

## Core Components

### Vulkan Compute Backend

The project leverages Vulkan 1.3+ with key extensions:
- **VK_KHR_cooperative_matrix** - Hardware-accelerated matrix operations
- **VK_KHR_external_memory_win32** - Cross-API memory sharing
- Buffer device addresses for efficient GPU memory access

### GEMM Shader

The `gemm.comp` shader implements a highly optimized General Matrix Multiply (GEMM) kernel:
- Tile-based computation with shared memory
- Prefetching for memory latency hiding
- Cooperative matrix operations at subgroup level
- Batch processing support

### Tensor Layout System

A flexible tensor layout abstraction that supports:
- Multi-dimensional shapes and strides
- Tiling for efficient GPU computation
- Layout composition and transformation
- Padding for bank conflict avoidance

## Requirements

- **Rust** 2024 edition or later
- **Vulkan** 1.3+ compatible GPU with:
  - Cooperative matrix support
  - Buffer device address feature
  - Shader float16/int8 support
- **Windows** (currently, due to Win32 external memory dependency)

## Building

```bash
# Clone the repository
git clone https://github.com/your-username/rosalia.git
cd rosalia

# Build the project
cargo build --release
```

## Shader Compilation

If you modify the compute shaders, recompile them using:

```bash
cd assets/shaders
compile.bat
```

This requires `glslangValidator.exe` to be in your PATH.

## Dependencies

| Crate       | Purpose                       |
| ----------- | ----------------------------- |
| vulkanalia  | Vulkan API bindings           |
| half        | FP16 arithmetic               |
| safetensors | Model weight loading          |
| bytemuck    | Safe memory transmutation     |
| tokio       | Async runtime                 |
| rayon       | Parallel iterators (optional) |

## Project Status

🚧 **Early Development** - This project is currently a work in progress. The core Vulkan infrastructure and GEMM kernels are functional, but the full RWKV inference pipeline is under active development.

## Related Projects

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) - Original RWKV implementation
- [RWKV-v8](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8) - RWKV-v8 with Rosa mechanism

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](http://opensource.org/licenses/MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Acknowledgments

- [BlinkDL](https://github.com/BlinkDL) for creating RWKV and the Rosa mechanism
