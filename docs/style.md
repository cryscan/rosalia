# Shader Coding Style

This document defines the coding conventions for GLSL compute shaders in this project.

## File Structure

Every shader file must follow this top-to-bottom order, with a blank line separating each section:

1. Version declaration and extensions
2. Pipeline constants (`layout(constant_id)`)
3. Compile-time constants (`const`)
4. Shared memory declarations (`shared`)
5. Buffer reference declarations (`layout(buffer_reference)`)
6. Params uniform block
7. Workgroup layout declaration
8. Helper functions
9. `main()` function

```glsl
// 1. Version & extensions
#version 450 core
#extension GL_KHR_shader_subgroup_basic : enable
...

// 2. Pipeline constants
layout(constant_id = 0) const uint C = 4096;    // Hidden size
...

// 3. Compile-time constants
const uint BLOCK_SIZE = 256;
...

// 4. Shared memory
shared I_TYPE sketch_sum[NUM_SUBGROUPS];
...

// 5. Buffer references
layout(buffer_reference) buffer Input { I_TYPE data[]; } input_x;
...

// 6. Params uniform block
layout(set = 0, binding = 0, std430) uniform Params {
    Input input_x;
    ...
} params;

// 7. Workgroup layout
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

// 8. Helper functions
uvec4 layout_iota(uvec4 shape, uint index) { ... }

// 9. main
void main() { ... }
```

## Naming Conventions

### Pipeline Constants (`layout(constant_id)`)

- **UPPER_SNAKE_CASE**
- Assigned sequential `constant_id` starting from 0
- Always provide a sensible default value
- Add an end-of-line comment describing the purpose

```glsl
layout(constant_id = 0) const uint C = 4096;    // Hidden size
layout(constant_id = 1) const uint T = 1;       // Number of tokens
layout(constant_id = 4) const float EPSILON = 1e-5;
```

### Compile-time Constants (`const`)

- **UPPER_SNAKE_CASE**
- Derived from pipeline constants or other compile-time constants
- Group logically; align the `=` sign within each group using extra spaces

```glsl
const uint BLOCK_SIZE       = 256;
const uint SUBGROUP_SIZE    = 32;
const uint NUM_SUBGROUPS    = BLOCK_SIZE / SUBGROUP_SIZE;
```

### Local Variables

- **lower_snake_case**
- Use `const` qualifier wherever possible

```glsl
const uint batch           = gl_WorkGroupID.y;
const uint token           = gl_WorkGroupID.x;
I_TYPE local_sum           = I_TYPE(0.0);
```

### Buffer Types and Variables

- **Buffer type**: PascalCase (`Input`, `Gamma`, `Output`)
- **Buffer variable**: lower_snake_case (`input_x`, `gamma`, `output_o`)
- The inner member is always named `data` and declared as an unsized array

```glsl
layout(buffer_reference) buffer InputA { uvec4 data[]; }  input_a;
layout(buffer_reference) buffer Gamma  { I_TYPE data[]; }  gamma;
layout(buffer_reference) buffer Output { C_TYPE data[]; }  output_o;
```

### Shared Memory

- **lower_snake_case**
- Prefix `sh_` for scratch / tile buffers used in tiled algorithms
- Other shared variables use descriptive names without prefix

```glsl
shared uvec4 sh_a[...];       // scratch buffer for A tiles
shared uvec4 sh_b[...];       // scratch buffer for B tiles
shared I_TYPE sketch_sum[8];  // reduction sketch
```

### Functions

- **lower_snake_case**
- Overloading is allowed when parameter types differ

```glsl
uvec4 layout_iota(uvec4 shape, uint index);
uvec2 layout_iota(uvec2 shape, uint index);
uint layout_index(uvec4 stride, uvec4 coord);
uint layout_index(uvec2 stride, uvec2 coord);
```

## Tensor SHAPE / STRIDE Convention

### Core Principle: Small Axis First

All SHAPE and STRIDE vectors follow the **small axis (innermost dimension) first** convention — the opposite of PyTorch / NumPy which place the large axis (outermost dimension, e.g. batch) first.

For a tensor with logical layout `[batch, token, hidden]`:

```glsl
// small axis first: (hidden, token, batch)
const uvec3 SHAPE_INPUT  = uvec3(C, T, 1);
const uvec3 STRIDE_INPUT = uvec3(STRIDE_INPUT_X, STRIDE_INPUT_Y, STRIDE_INPUT_Z);
```

The equivalent PyTorch layout would be `shape = (1, T, C)`.

### SHAPE Naming

- **UPPER_SNAKE_CASE**
- Pattern: `SHAPE_<TENSOR>`
- The vector dimension matches the rank of the tensor
- Use `uvec2`, `uvec3`, or `uvec4` as appropriate

```glsl
const uvec3 SHAPE_INPUT      = uvec3(C, T, 1);                          // rank 3
const uvec4 SHAPE_A_TILE     = uvec4(TILE_K, TILE_M, NUM_TILE_K, NUM_TILE_M);  // rank 4
const uvec2 SHAPE_SH_A_TILE  = uvec2(TILE_K, TILE_M);                   // rank 2
```

### STRIDE Naming

- **UPPER_SNAKE_CASE**
- Pattern: `STRIDE_<TENSOR>_<COMPONENT>` for individual stride constants
  - Components use `X`, `Y`, `Z`, `W`
  - `_X` is always declared explicitly even when its value is 1
- Pattern: `STRIDE_<TENSOR>` for the composed stride vector
- Individual stride constants are declared as pipeline constants; the composed vector is a `const`

```glsl
// Pipeline constants for individual strides
layout(constant_id = 2) const uint STRIDE_INPUT_X = 1;
layout(constant_id = 3) const uint STRIDE_INPUT_Y = 4096;
layout(constant_id = 4) const uint STRIDE_INPUT_Z = 4096;

// Composed stride vector
const uvec3 STRIDE_INPUT = uvec3(STRIDE_INPUT_X, STRIDE_INPUT_Y, STRIDE_INPUT_Z);
```

For higher-rank tensors:

```glsl
layout(constant_id = 9)  const uint STRIDE_A_TILE_X = 1;
layout(constant_id = 10) const uint STRIDE_A_TILE_Y = 4096;
layout(constant_id = 11) const uint STRIDE_A_TILE_Z = 128;
layout(constant_id = 12) const uint STRIDE_A_TILE_W = 131072;

const uvec4 STRIDE_A_TILE = uvec4(STRIDE_A_TILE_X, STRIDE_A_TILE_Y, STRIDE_A_TILE_Z, STRIDE_A_TILE_W);
```

### Component Mapping

| Vector Component | Axis Position | Stride Value (contiguous) | Constant Name Suffix |
|------------------|---------------|---------------------------|----------------------|
| `.x`             | 0 (innermost) | 1                         | `_X`                 |
| `.y`             | 1             | shape[0] × stride.x       | `_Y`                 |
| `.z`             | 2             | shape[0] × stride.y       | `_Z`                 |
| `.w`             | 3             | shape[0] × stride.y × stride.z | `_W`            |

### Index Computation

Given SHAPE and STRIDE, convert between linear and multi-dimensional indices:

```glsl
// Multi-dim → linear
uint index = stride.x * coord.x + stride.y * coord.y + stride.z * coord.z + stride.w * coord.w;

// Linear → multi-dim (layout_iota)
uvec4 coord;
coord.x = index % shape.x;
coord.y = (index / shape.x) % shape.y;
coord.z = (index / (shape.x * shape.y)) % shape.z;
coord.w = index / (shape.x * shape.y * shape.z);
```

## `#define` and Conditional Compilation

### Macro Definitions

- **UPPER_SNAKE_CASE** for macro names
- Parenthesize all parameters and the entire expression body
- Used sparingly; prefer `const` when possible

```glsl
#define DIV_CEIL(A, B)  ((A) + (B) - 1) / (B)
```

### Feature Flags

- **UPPER_SNAKE_CASE** (e.g. `AFFINE`, `ACTIVATION`)
- Tested with `#if defined(FLAG)` for boolean flags
- Numeric flags tested with `#if FLAG == N`

```glsl
#if defined(AFFINE)
    ...
#endif

#if defined(ACTIVATION)
    #if ACTIVATION == 1
        val = max(val, C_TYPE(0));
        val = val * val;
    #endif
    #if ACTIVATION == 2
        val = tanh(val);
    #endif
#endif
```

## Formatting

### Alignment

- Align the `=` sign within groups of related declarations using extra spaces

```glsl
const uint BLOCK_SIZE       = 256;
const uint SUBGROUP_SIZE    = 32;
const uint NUM_SUBGROUPS    = BLOCK_SIZE / SUBGROUP_SIZE;
```

### Attributes

- Place loop attributes on the line immediately before the loop
- Use `[[unroll]]` for all loops with known trip counts

```glsl
[[unroll]]
for (uint i = index; i < SHAPE_INPUT.x; i += BLOCK_SIZE) {
    ...
}
```

### Blank Lines

- One blank line between logical sections
- No blank lines within a tightly coupled block (e.g. between pipeline constant declarations)

### Indentation

- 4 spaces per level (no tabs)

### Braces

- Opening brace on the same line as the control statement
- Closing brace on its own line

```glsl
if (subgroup_invocation_id == 0) {
    sketch_sum[subgroup_id] = local_sum;
}
```
