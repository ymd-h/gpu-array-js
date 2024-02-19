# Miscellaneous Information

## WebGPU Basics
### Use float 16 on WebGPU

Check whether GPU supports `shader-f16` feature;

```javascript
const adapter = await navigator.gpu.requestAdapter();
adapter.features.has("shader-f16"); // -> true / false
```

Create (Logical) GPU Device with `shader-f16` feature;

```javascript
const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
```

Enable `f16` extension at WebGPU Shading Language (WGSL);

```wgsl
enable f16;
...
```


### `GPUBuffer.mapAsync` automaticall wait previous job


```javascript
// await device.queue.onSubmittedWorkDone() <- Not necessary
await buffer.mapAsync(GPUMapMode.READ, 0, size);
```
