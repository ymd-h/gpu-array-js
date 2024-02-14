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

- `GPUBackend.add(lhs: NDArray, rhs: NDArray, out: NDArray?): NDArray`
- `GPUBackend.sub(lhs: NDArray, rhs: NDArray, out: NDArray?): NDArray`
- `GPUBackend.mul(lhs: NDArray, rhs: NDArray, out: NDArray?): NDArray`
- `GPUBackend.div(lhs: NDArray, rhs: NDArray, out: NDArray?): NDArray`
- `GPUBackend.abs(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.acos(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.acosh(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.asin(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.asinh(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.atan(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.atanh(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.atan2(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.ceil(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.clamp(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.cos(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.cosh(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.exp(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.exp2(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.floor(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.log(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.log2(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.sign(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.sin(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.sinh(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.sqrt(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.tan(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.tanh(arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend.max(arg0: NDArray, arg1: NDArray, out: NDArray?): NDArray`
- `GPUBackend.min(arg0: NDArray, arg1: NDArray, out: NDArray?): NDArray`
- `GPUBackend.pow(arg0: NDArray, arg1: NDArray, out: NDArray?): NDArray`


### Custom Function for WGSL Built-in Function
We don't predefine all the WGSL built-in functions,
but you can still use them.

cf. [WGSL Numeric Built-in Functions](https://gpuweb.github.io/gpuweb/wgsl/#numeric-builtin-functions)

- `GPUBackend._func1(f: string, arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend._func2(f: string, arg0: NDArray, arg1: NDArray, out: NDArray?): NDArray`

`f` is a built-in function name.


### Custom Function from Scratch
> [!WARNING]
> This API is not user friendly, nor intended to use.

- `GPUBackend.createShader(code: string): GPUShaderModule`
- `GPUBackend.execute(shader: GPUShaderModule, specs: {array: NDArray, mode: "read-only" | "write-only" | "read-write"}[], dispatch: number[]): undefined`


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
