# GPU Array

> [!CAUTION]
> This repository is under development, and might not work.

This JavaScript PoC code provides NDArray
(aka. multidimensional array) based on WebGPU.


> [!WARNING]
> Only limited browsers and environments support WebGPU.
> cf. [WebGPU API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#browser_compatibility)



## Example Usage

```javascript
import { createGPU } from "https://cdn.jsdelivr.net/gh/ymd-h/gpu-array-js/gpu-array.js";

// 1. Create GPU. (Throw Error, if WebGPU is not available)
const gpu = await createGPU();

// 2. Create NDArray
const a = gpu.Array({ shape: [2, 2] });
const b = gpu.Array({ shape: [2, 2] });

// 3. Set Data
a.set(1, 0, 0); // Set 1 at (0, 0)
a.set(1, 1, 1); // Set 1 at (1, 1)

b.set(2, 0, 1); // Set 2 at (0, 1)
b.set(3, 1, 1); // Set 3 at (1, 1)

// 4. Execute Calculation.
// (If data is updated, automatically send to GPU)
const c = gpu.add(a, b); // c = a + b

// Optional: You can send data manually.
// a.send();


// 5. Get Data
// (If gpu data is updated, automatically load from GPU)
console.log(await c.get(0, 0));
console.log(await c.get(0, 1));
console.log(await c.get(1, 0));
console.log(await c.get(1, 1));

// Optional: You can load data manually.
// await c.load();
// console.log(c.get_without_load(0, 0));
// console.log(c.get_without_load(0, 1));
// console.log(c.get_without_load(1, 0));
// console.log(c.get_without_load(1, 1));
```

## Features
### Predefined Functions

- `GPUBackend.add(lhs: GPUArray, rhs: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.sub(lhs: GPUArray, rhs: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.mul(lhs: GPUArray, rhs: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.div(lhs: GPUArray, rhs: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.abs(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.acos(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.acosh(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.asin(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.asinh(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.atan(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.atanh(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.atan2(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.ceil(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.clamp(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.cos(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.cosh(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.exp(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.exp2(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.floor(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.log(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.log2(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.sign(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.sin(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.sinh(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.sqrt(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.tan(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.tanh(arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.max(arg0: GPUArray, arg1: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.min(arg0: GPUArray, arg1: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend.pow(arg0: GPUArray, arg1: GPUArray, out: GPUArray?): GPUArray`


### Custom Function for WGSL Built-in Function
We don't predefine all the WGSL built-in functions,
but you can still use them.

cf. [WGSL Numeric Built-in Functions](https://gpuweb.github.io/gpuweb/wgsl/#numeric-builtin-functions)

- `GPUBackend._func1(f: string, arg: GPUArray, out: GPUArray?): GPUArray`
- `GPUBackend._func2(f: string, arg0: GPUArray, arg1: GPUArray, out: GPUArray?): GPUArray`

`f` is a built-in function name.


### Custom Function from Scratch
> [!WARNING]
> This API is not user friendly, nor intended to use.

- `GPUBackend.createShader(code: string): GPUShaderModule`
- `GPUBackend.execute(shader: GPUShaderModule, specs: {array: GPUArray, mode: "read-only" | "write-only" | "read-write"}[], dispatch: number[]): undefined`


`dispatch` are number of GPU workgroups of X, Y, Z. 1 <= `dispatch.length` <= 3.

## Design

### Template-based Shader
In my [previous work](https://github.com/ymd-h/vulkpy),
shader management was [one of the biggest problems](https://github.com/ymd-h/vulkpy/issues/2).
Here, we try to implement type agnostic compute shaders in [shader.js](https://github.com/ymd-h/gpu-array-js/blob/master/shader.js).
Types are passed as arguments.
Moreover, similar computations (e.g. `a + b` and `a - b`, etc.) are generated
from single template.


### Update Tracking
CPU-side and GPU-side data updates are tracked with `.cpu_dirty` and `.gpu_dirty`
properties of `NDArray`.
Only when the data are updated `send()` / `load()` methods acutually copy data.



## Limitations

A lot of features still missing;

- Broadcasting
- Non Element-wise Computation (e.g. Matrix Multiplication)
- Non Single Element `get()` / `set()`
- `f16` (supported by WebGPU, but no corresponding `TypedArray`)
- Custom Strides
