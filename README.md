# gpu-array.js: NDArray on WebGPU

> [!CAUTION]
> This repository is under development, and might not work.

This JavaScript PoC code provides NDArray
(aka. multidimensional array) based on WebGPU.


> [!WARNING]
> Only limited browsers and environments support WebGPU.
> cf. [WebGPU API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#browser_compatibility)



## 1. Example Usage

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

## 2. API
### 2.1 Types
- `@typedef {Object} AdapterOptions`
  - `@property {"low-power" | "high-performance" | "undefined"} powerPreference`
- `@typedef {Object} DeviceOptions`
  - `@property {{label: string} | undefined} defaultQueue`
  - `@property {string?} label`
  - `@property {string[] | undefined} requiredFeatures`
  - `@property {Object.<string, *>} requiredLimits`
- `@typedef {Object} GPUOptions`
  - `@property {AdapterOptions?} adapter`
  - `@property {DeviceOptions?} device`
- `@typedef {"i32" | "u32" | "f16" | "f32"} DType`
- `@typedef {Object} ArrayOptions`
  - `@property {number | number[] | undefined} shape`
  - `@property {Dtype?} dtype`
  - `@property {number | number[] | undefined} strides`
- `@typedef {Object} RangeOptions`
  - `@property {number?} start`
  - `@property {number} stop`
  - `@property {number?} step`
- `@typedef {Object} PRNGOptions`
  - `@property {number | bigint | undefined} seed`
  - `@property {number?} size`


### 2.2 Exported (Free) Function
- `createGPU(options: GPUOptions?): Promise<GPUBackend>`


### 2.3 Array Creation
- `GPUBackend.Array(options: ArrayOptions?): NDArray`
- `GPUBackend.ones(options: ArrayOptions?): NDArray`
- `GPUBackend.full(value: number, options: ArrayOptions?): NDArray`
- `GPUBackend.arange(range: RangeOptions, options: ArrayOptions?): NDArray`

### 2.4 Array Method
- `NDArray.get(...index: number[]): Promise<number>`
- `NDArray.get_without_load(...index: number[]): number`
- `NDArray.set(value: number | number[] | TypedArray, ...index: number[]): undefined`
- `NDArray.load(): Promise<undefined>`
- `NDArray.send(): undefined`


### 2.5 Predefined Functions
#### 2.5.1 Element-wise (Support Broadcast)

- `GPUBackend.add(lhs: NDArray | number, rhs: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.sub(lhs: NDArray | number, rhs: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.mul(lhs: NDArray | number, rhs: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.div(lhs: NDArray | number, rhs: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.abs(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.acos(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.acosh(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.asin(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.asinh(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.atan(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.atanh(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.atan2(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.ceil(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.clamp(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.cos(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.cosh(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.exp(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.exp2(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.floor(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.log(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.log2(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.sign(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.sin(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.sinh(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.sqrt(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.tan(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.tanh(arg: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.max(arg0: NDArray | number, arg1: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.min(arg0: NDArray | number, arg1: NDArray | number, out: NDArray?): NDArray`
- `GPUBackend.pow(arg0: NDArray | number, arg1: NDArray | number, out: NDArray?): NDArray`

#### 2.5.2 Reduction

- `GPUBackend.sum(arg: NDArray): NDArray`
- `GPUBackend.prod(arg: NDArray): NDArray`
- `GPUBackend.minimum(arg: NDArray): NDArray`
- `GPUBackend.maximum(arg: NDArray): NDArray`


### 2.6 Custom Element-wise Function for WGSL Built-in Function
We don't predefine all the WGSL built-in functions,
but you can still use them.

cf. [WGSL Numeric Built-in Functions](https://gpuweb.github.io/gpuweb/wgsl/#numeric-builtin-functions)

- `GPUBackend._func1(f: string, arg: NDArray, out: NDArray?): NDArray`
- `GPUBackend._func2(f: string, arg0: NDArray, arg1: NDArray, out: NDArray?): NDArray`

`f` is a built-in function name.


### 2.7 Custom Function from Scratch
> [!WARNING]
> This API is not user friendly, nor intended to use.

- `GPUBackend.createShader(code: string): GPUShaderModule`
- `GPUBackend.execute(shader: GPUShaderModule, specs: {array: NDArray, mode: "read-only" | "write-only" | "read-write"}[], dispatch: number[]): undefined`


`dispatch` are number of GPU workgroups of X, Y, Z. 1 <= `dispatch.length` <= 3.

### 2.8 Pseudo Random Number Generator (PRNG)
- `GPUBackend.Xoshiro128pp(options: PRNGOptions?): Xoshiro128pp`
- `Xoshiro128pp.next(dtype: "u32" | "f32"): NDArray`
- `Xoshiro128pp.normal(dtype: "f32" | "f16"): NDArray`


## 3. Design

### 3.1 Template-based Shader
In my [previous work](https://github.com/ymd-h/vulkpy),
shader management was [one of the biggest problems](https://github.com/ymd-h/vulkpy/issues/2).
Here, we try to implement type agnostic compute shaders in [shader.js](https://github.com/ymd-h/gpu-array-js/blob/master/shader.js).
Types are passed as arguments.
Moreover, similar computations (e.g. `a + b` and `a - b`, etc.) are generated
from single template.


### 3.2 Update Tracking
CPU-side and GPU-side data updates are tracked with `.cpu_dirty` and `.gpu_dirty`
properties of `NDArray`.
Only when the data are updated `send()` / `load()` methods acutually copy data.



## 4. Limitations

A lot of features are still missing;

- Linear Algebra (e.g. Matrix Multiplication)


The size of data must be multiple of 4 bytes,
so that "f16" `NDArray` must have even elements.


## 5. Dependencies
- [petamoriken/float16](https://github.com/petamoriken/float16) (MIT License)
  - For `Float16Array`. Once [this proposal](https://github.com/tc39/proposal-float16array) (Stage 3) become available, we will replace.


## 6. Notes

`"f16"` is supported only when GPU supports it.
Inside `createGPU()` function, we check its supported features,
and automatically add `"shader-f16"` to `requiredFeatures` if possible.
