/** @module gpu-array */

import {
    vector_op, vector_op_indirect,
    func1,
    func2, func2_indirect,
    reduce_op, reduce_func,
} from "./shader.js";


/**
 * @typedef {Object} AdapterOptions
 * @property {"low-power" | "high-performance" | undefined} powerPreference
 *
 * @typedef {Object} DeviceOptions
 * @property {{label: string} | undefined} defaultQueue
 * @property {string?} label
 * @property {string[] | undefined} requiredFeatures
 * @property {Object.<string, *>} requiredLimits
 *
 * @typedef {Object} GPUOptions
 * @property {AdapterOptions?} adapter
 * @property {DeviceOptions?} device
 *
 * @typedef {"i32" | "u32" | "f16" | "f32"} DType
 *
 * @typedef {"read-only" | "write-only" | "read-wirite"} Mode
 *
 * @typedef {Object} ArraySpec
 * @property {NDArray} array
 * @property {Mode} mode
 *
 * @typedef {Object} ArrayOptions
 * @property {number | number[] | undefined} shape
 * @property {DType?} dtype
 * @property {number | number[] | undefined} strides
 *
 * @typedef {Object} RangeOptions
 * @property {number?} start
 * @property {number} stop
 * @property {number?} step
 *
 * @typedef {Object} Layout
 * @property {GPUBindGroupLayout} bindGroup
 * @property {GPUPipelineLayout} pipeline
 */


/**
 * @param {string} t1
 * @param {string} t2
 * @returns {string}
 */
const promoteType = (t1, t2) => {
    if((t1 === "f32") || (t2 === "f32")){
        return "f32";
    }

    if((t1 === "f16") || (t2 === "f16")){
        return "f16";
    }

    throw new Error(`Incompatible Types: ${t1}, ${t2}`);
}


/**
 * @param {number[][]} shapes
 * @returns {number[]}
 */
const broadcastShapes = (...shapes) => {
    const length = shapes.reduce((a, s) => Math.max(a, s.length), 0);

    return Array.from(
        { length },
        (_, i) => shapes.reduce(
            (a, s) => {
                if(i < length - s.length){ return a; }
                const si = s[i - (length - s.length)];

                if((a === si) || (si === 1)){ return a; }
                if(a === 1){ return si; }

                throw new Error(`Incompatible Shape`);
            },
            1
        ),
    );
};


/**
 * @param {NDArray} array
 * @param {number[]} shape
 * @returns {number[]}
 */
const broadcastStrides = (array, shape) => {
    const strides = [...array.strides];

    if(strides.length > shape.length){
        const s1 = array.shape.join(",");
        const s2 = shape.join(",");
        throw new Error(`Incompatible Shape: [${s1}] -> [${s2}]`);
    }

    while(strides.length < shape.length){
        strides.unshift(0);
    }
    const diff = shape.length - array.shape.length; // [0, shape.length)

    for(let i = 0; i < shape.length; i++){
        if((strides[i] === 0) || (array.shape[i - diff] === shape[i])){
            continue;
        }
        if(array.shape[i - diff] !== 1){
            const s1 = array.shape.join(",");
            const s2 = shape.join(",");
            throw new Error(`Incompatible Shape: [${s1}] -> [${s2}]`);
        }
        strides[i] = 0;
    }

    return strides;
};


class GPUBackend {
    /**
     * @constructor
     * @param {GPUDevice}
     */
    constructor(device){
        /** @type {GPUDevice} */
        this.device = device;

        /** @type {GPUDeviceLostInfo?} */
        this.lost = null;
        this.device.lost.then((lost) => {
            this.lost = lost;
        });

        /** @type {Map<string, Layout>} */
        this.layout = new Map();

        /** @type {Map<string, GPUShaderModule>} */
        this.shader = new Map();

        // Vector Operand
        const vop = [
            ["add", "+"],
            ["sub", "-"],
            ["mul", "*"],
            ["div", "/"],
        ];
        for(const [name, op] of vop){
            this[name] = (lhs, rhs, out) => this._vector_op(op, lhs, rhs);
        }

        // Function with 1 Argument
        const f1 = [
            "abs",
            "acos", "acosh",
            "asin", "asinh",
            "atan", "atanh", "atan2",
            "ceil",
            "clamp",
            "cos", "cosh",
            "exp", "exp2",
            "floor",
            "log", "log2",
            "sign",
            "sin", "sinh",
            "sqrt",
            "tan", "tanh",
        ];
        for(const f of f1){
            this[f] = (arg, out) => this._func1(f, arg);
        }

        // Function with 2 Arguments
        const f2 = [
            "max", "min",
            "pow",
        ];
        for(const f of f2){
            this[f] = (arg0, arg1, out) => this._func2(f, arg0, arg1, out);
        }

        // Reduction Op
        const red_op = [
            ["sum", "+"],
            ["prod", "*"],
        ];
        for(const [name, op] of red_op){
            this[name] = (arg) => this._reduce_op(op, arg);
        }

        const red_f = [
            ["minimum", "min"],
            ["maximum", "max"],
        ];
        for(const [name, f] of red_f){
            this[name] = (arg) => this._reduce_func(f, arg);
        }
    }

    assertLost(){
        if(this.lost !== null){
            const { message, reason } = this.lost;
            const msg = `GPU has been lost: { message: ${message}, reason: ${reason} }`;
            throw new Error(msg);
        }
    }

    /**
     * Create Layout
     * @param {Mode[]} modes
     * @returns {Layout}
     */
    createLayout(modes){
        this.assertLost();

        const key = modes.map(m => (m === "read-only") ? "r" : "w").join("");

        if(this.layout.has(key)){
            return this.layout.get(key);
        }

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: Array.from(modes, (m, i) => {
                return {
                    binding: i,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: (m === "read-only") ? "read-only-storage" : "storage",
                    },
                };
            }),
            label: `BindGroupLayout-${key}`,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
            label: `PipelineLayout-${key}`,
        });

        const L = { bindGroupLayout, pipelineLayout };
        this.layout.set(key, L);

        return L;
    }

    /**
     * Create Shader
     * @param {string} code
     * @returns {GPUShaderModule}
     */
    createShader(code){
        this.assertLost();

        if(this.shader.has(code)){
            return this.shader.get(code);
        }

        const shader = this.device.createShaderModule({ code });
        shader.getCompilationInfo().then(({ messages }) => {
            for(const { length, lineNum, linePos, message, offset, type } of messages){
                const m = (
                    lineNum ?
                        `Line ${lineNum}:${linePos} - ${code.substr(offset,length)}\n` :
                        ""
                ) + message;

                switch(type){
                case "error":
                    throw new Error(m);
                    break;
                case "warning":
                    console.warn(m);
                    break;
                case "info":
                    console.log(m);
                    break;
                }
            }
        });

        this.shader.set(code, shader);

        return shader;
    }

    /**
     * Create NDArray
     * @param {ArrayOptions?} options
     * @returns {NDArray}
     */
    Array(options){
        this.assertLost();
        return new NDArray(this.device, options);
    }

    /**
     * Create NDArray filled with 1
     * @param {ArrayOptions?} options
     * @returns {NDArray}
     */
    ones(options){
        const a = this.Array(options);
        a.cpu.fill(1);
        a.cpu_dirty = true;
        return a;
    }

    /**
     * Create NDArray filled with value
     * @param {number} value
     * @param {ArrayOptions?} options
     * @returns {NDArray}
     */
    full(value, options){
        const a = this.Array(options);
        a.cpu.fill(value);
        a.cpu_dirty = true;
        return a;
    }

    /**
     * Create Range NDArray
     * @param {RangeOptions} range
     * @param {ArrayOptions?} options
     */
    arange({ start, stop, step }, options){
        if(stop === undefined){
            throw new Error(`stop is required`);
        }

        start ??= 0;

        step ??= 1;
        if(step === 0){
            throw new Error(`step === 0 is not allowed`);
        }

        const cond = step > 0 ? (v => v < stop) : (v => v > stop);

        const range = [];
        for(let i = 0; true; i++){
            const v = start + i * step;
            if(!cond(v)){ break; }

            range.push(v);
        }

        const a = this.Array({ dtype: "i32", shape: range.length, ...options });
        if(range.length !== a.length){
            throw new Error(`Incompatible shape`);
        }

        a.cpu.set(range);
        a.cpu_dirty = true;
        return a;
    }

    #stridesBuffer(strides){
        this.assertLost();
        const buffer = this.device.createBuffer({
            size: 4 * strides.length,
            usage: GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, Uint32Array.from(strides));
        return {
            send: () => {},
            gpu: buffer,
        };
    }

    /**
     * Execute GPU Computation
     * @param {GPUShaderModule} shader
     * @param {ArraySpec[]} specs
     * @param {number[]} dispatch
     */
    execute(shader, specs, dispatch){
        for(const {array, mode} of specs){
            switch(mode){
            case "read-only":
                array.send();
                break;
            case "write-only":
                array.cpu_dirty = false;
                array.gpu_dirty = true;
                break;
            case "read-write":
                array.send();
                array.gpu_dirty = true;
                break;
            default:
                throw new Error(`Unknown mode: ${mode}`);
            }
        }

        const {
            bindGroupLayout, pipelineLayout,
        } = this.createLayout(specs.map(s => s.mode));

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: specs.map((s, i) => {
                return {
                    binding: i,
                    resource: { buffer: s.array.gpu },
                };
            }),
        });

        const pipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: "main",
            },
        });

        const cmd = this.device.createCommandEncoder();
        const pass = cmd.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...dispatch);
        pass.end();
        this.device.queue.submit([cmd.finish()]);
    }


    _vector_op(op, lhs, rhs, out){
        const dtype = (lhs.dtype === rhs.dtype) ?
              lhs.dtype :
              promoteType(lhs.dtype, rhs.dtype);

        out ??= this.Array({ shape: broadcastShapes(lhs.shape, rhs.shape), dtype });
        const size = this.device.limits.maxComputeWorkgroupSizeX;

        if(out.custom_strides){
            throw new Error(`Custom Strides for out is not supported`);
        }

        const shader_args = [
                op, size,
                {binding: 0, type: lhs.dtype, conv: (dtype === lhs.dtype) ? "" : dtype},
                {binding: 1, type: rhs.dtype, conv: (dtype === rhs.dtype) ? "" : dtype},
                {binding: 2, type: out.dtype, conv: (dtype === out.dtype) ? "" : dtype},
        ];

        const execute_buffers = [
                {array: lhs, mode: "read-only"},
                {array: rhs, mode: "read-only"},
                {array: out, mode: "write-only"},
        ];

        let lhs_strides = null;
        let rhs_strides = null;
        let out_strides = null;

        const use_strides = (
            lhs.custom_strides || rhs.custom_strides ||
                (out.shape.length !== lhs.shape.length) ||
                (out.shape.length !== rhs.shape.length) ||
                lhs.shape.some((s, i) => s !== out.shape[i]) ||
                rhs.shape.some((s, i) => s !== out.shape[i])
        );
        if(use_strides){
            lhs_strides = this.#stridesBuffer(broadcastStrides(lhs, out.shape));
            rhs_strides = this.#stridesBuffer(broadcastStrides(rhs, out.shape));
            out_strides = this.#stridesBuffer(out.strides);

            shader_args.push(
                {binding: 3},
                {binding: 4},
                {binding: 5},
            );

            execute_buffers.push(
                {array: lhs_strides, mode: "read-only"},
                {array: rhs_strides, mode: "read-only"},
                {array: out_strides, mode: "read-only"},
            );
        }

        const shader = this.createShader(
            use_strides ?
                vector_op_indirect(...shader_args):
                vector_op(...shader_args),
        );

        this.execute(shader, execute_buffers, [Math.ceil(out.length / size)]);
        this.device.queue.onSubmittedWorkDone().then(() => {
            lhs_strides?.gpu.destroy();
            rhs_strides?.gpu.destroy();
            out_strides?.gpu.destroy();
        });
        return out;
    }

    _func1(f, arg, out){
        out ??= this.Array({ shape: arg.shape, dtype: arg.dtype });
        const size = this.device.limits.maxComputeWorkgroupSizeX;

        if(out.custom_strides){
            throw new Error(`Custom Strides for out is not supported`);
        }

        const out_conv = (arg.dtype === out.dtype) ? "" : out.dtype;

        if(arg.custom_strides ||
           arg.shape.length !== out.shape.length ||
           arg.shape.some((s, i) => s !== out.shape[i])){
            throw new Error(`Broadcast is not supported`);
        }

        const shader = this.createShader(
            func1(
                f, size,
                {binding: 0, type: arg.dtype, conv: ""},
                {binding: 1, type: out.dtype, conv: out_conv},
            ),
        );

        this.execute(
            shader,
            [
                {array: arg, mode: "read-only"},
                {array: out, mode: "write-only"},
            ],
            [Math.ceil(out.length / size)],
        );

        return out;
    }

    _func2(f, arg0, arg1, out){
        const dtype = (arg0.dtype === arg1.dtype) ?
              arg0.dtype :
              promoteType(arg0.dtype, arg1.dtype);

        out ??= this.Array({ shape: broadcastShapes(arg0.shape, arg1.shape), dtype });
        const size = this.device.limits.maxComputeWorkgroupSizeX;

        if(out.custom_strides){
            throw new Error(`Custom Strides for out is not supported`);
        }

        const conv = (v) => (v.dtype === dtype) ? "" : dtype;

        const use_strides = (arg0.custom_strides ||
                             arg1.custom_strides ||
                             arg0.shape.length !== out.shape.length ||
                             arg1.shape.length !== out.shape.length ||
                             arg0.shape.some((s, i) => s !== out.shape[i]) ||
                             arg1.shape.some((s, i) => s !== out.shape[i]));

        const shader_args = [
            f, size,
            [
                { binding: 0, type: arg0.dtype, conv: conv(arg0) },
                { binding: 1, type: arg1.dtype, conv: conv(arg1) },
            ],
            { binding: 2, type: out.dtype, conv: conv(out) },
        ];

        const execute_buffers = [
                {array: arg0, mode: "read-only"},
                {array: arg1, mode: "read-only"},
                {array: out, mode: "write-only"},
        ];

        let arg0_strides = null;
        let arg1_strides = null;
        let out_strides = null;

        if(use_strides){
            arg0_strides = this.#stridesBuffer(broadcastStrides(arg0, out.shape));
            arg1_strides = this.#stridesBuffer(broadcastStrides(arg1, out.shape));
            out_strides = this.#stridesBuffer(out.strides);

            shader_args.push(
                {binding: 3},
                {binding: 4},
                {binding: 5},
            );

            execute_buffers.push(
                {array: arg0_strides, mode: "read-only"},
                {array: arg1_strides, mode: "read-only"},
                {array: out_strides, mode: "read-only"},
            );
        }

        const shader = this.createShader(
            use_strides ?
                func2_indirect(...shader_args) :
                func2(...shader_args),
        );

        this.execute(shader, execute_buffers, [Math.ceil(out.length / size)]);
        this.device.queue.onSubmittedWorkDone().then(() => {
            arg0_strides?.gpu.destroy();
            arg1_strides?.gpu.destroy();
            out_strides?.gpu.destroy();
        });
        return out;
    }

    _reduce_op(op, arg){
        if(arg.custom_strides){
            throw new Error(`Reduce Op hasn't supported custom strides yet`);
        }

        while(true){
            const length = (arg.length > 64) ?
                  (1 << (Math.floor(Math.log2(arg.length)) -1)):
                  1;
            const out = this.Array({ shape: length, dtype: arg.dtype });

            const shader = this.createShader(
                reduce_op(
                    op, length,
                    {binding: 0, type: arg.dtype, conv: ""},
                    {binding: 1, type: out.dtype, conv: ""},
                ),
            );

            this.execute(
                shader,
                [
                    {array: arg, mode: "read-only"},
                    {array: out, mode: "write-only"},
                ],
                [1],
            );

            if(length === 1){
                return out;
            }

            arg = out;
        }
    }

    _reduce_func(f, arg){
        if(arg.custom_strides){
            throw new Error(`Reduce Func hasn't supported custom strides yet`);
        }

        while(true){
            const length = (arg.length > 64) ?
                  (1 << (Math.floor(Math.log2(arg.length)) -1)):
                  1;
            const out = this.Array({ shape: length, dtype: arg.dtype });

            const shader = this.createShader(
                reduce_func(
                    f, length,
                    {binding: 0, type: arg.dtype, conv: ""},
                    {binding: 1, type: out.dtype, conv: ""},
                ),
            );

            this.execute(
                shader,
                [
                    {array: arg, mode: "read-only"},
                    {array: out, mode: "write-only"},
                ],
                [1],
            );

            if(length === 1){
                return out;
            }

            arg = out;
        }
    }
};



class NDArray {
    /** @type {Promise<undefined>?} */
    #load_promise

    /**
     * @constructor
     * @param {GPUDevice} device
     * @param {ArrayOptions?} options
     */
    constructor(device, options){
        /** @type {GPUDevice} */
        this.device = device;

        /** @type {bool} */
        this.cpu_dirty = false;

        /** @type {bool} */
        this.gpu_dirty = false;

        let { shape, dtype, strides } = options ?? {};

        // shape
        this.shape = this.#ensure_shape(shape);

        // strides
        /** @type {bool} */
        this.custom_strides = (strides !== undefined);

        strides ??= this.shape.reduce((a, si) => {
            a = a.map(ai => ai * si);
            a.push(1);
            return a;
        }, []);
        if(typeof strides === "number"){
            strides = [strides];
        }
        if(strides.length !== this.shape.length){
            throw new Error(`strides must have same length: ${strides.length}`);
        }
        this.strides = strides.map(s => s | 0);

        // dtype
        /** @type {str} */
        this.dtype = dtype ?? "f32";

        /** @type {number} */
        this.itemsize = 4;

        /** @type {ArrayLike} */
        this.cpu;

        /** @type {number} */
        this.length = this.custom_strides ?
            this.shape.reduce((a, v, i) => a + (v-1)*this.strides[i], 1) :
            this.shape.reduce((a, v) => a * v, 1);

        switch(this.dtype){
        case "f16":
            this.itemsize = 2;
            throw new Error(`f16 hasn't been supported yet`);
            break;
        case "f32":
            this.cpu = new Float32Array(this.length);
            break;
        case "u32":
            this.cpu = new Uint32Array(this.length);
            break;
        case "i32":
            this.cpu = new Int32Array(this.length);
            break;
        default:
            throw new Error(`Unknown dtype: ${dtype}`);
        }

        /** @type {GPUBuffer} */
        this.gpu = this.device.createBuffer({
            size: this.length * this.itemsize,
            usage: GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
        });

        this.#load_promise = null;
    }

    #ensure_shape(shape){
        shape ??= [1];
        if(typeof shape === "number"){
            shape = [shape];
        }
        if(shape.some(s => s <= 0)){
            throw new Error(`shape must be positive: [${shape.join()}]`);
        }
        return shape.map(s => s | 0);
    }

    /**
     * Reshape
     * @param {number | number[] | undefined} shape
     */
    reshape(shape){
        if(this.custom_strides){
            throw new Error(`reshape() is not supported with custom strides`);
        }
        shape = this.#ensure_shape(shape);

        if(shape.reduce((a, s) => a * s, 1) !== this.length){
            const s1 = this.shape.join(",");
            const s2 = shape.join(",");
            throw new Error(`Reshape with incompatible shape: [${s1}] -> [${s2}]`);
        }
        this.shape = shape;
    }

    /**
     * Load Data from GPU if necessary
     * @returns {Promise<undefined>}
     */
    load(){
        if(!this.gpu_dirty){
            return;
        }

        if(this.#load_promise !== null){
            return this.#load_promise;
        }

        const staging = this.device.createBuffer({
            size: this.gpu.size,
            usage: GPUBufferUsage.MAP_READ |
                GPUBufferUsage.COPY_DST,
        });

        const cmd = this.device.createCommandEncoder();
        cmd.copyBufferToBuffer(this.gpu, 0, staging, 0, this.gpu.size);
        this.device.queue.submit([cmd.finish()]);

        // Note: mapAsync ensures submitted work done before.
        this.#load_promise = staging.mapAsync(GPUMapMode.READ, 0, this.gpu.size).then(
            () => {
                const b = new Uint8Array(staging.getMappedRange(0, this.gpu.size));
                (new Uint8Array(this.cpu.buffer)).set(b);

                staging.unmap();
                staging.destroy();

                this.gpu_dirty = false;
                this.#load_promise = null;
            }
        );

        return this.#load_promise;
    }

    /**
     * Send Data to GPU if necessary
     */
    send(){
        if(!this.cpu_dirty){
            return;
        }

        this.device.queue.writeBuffer(this.gpu, 0, this.cpu);
        this.cpu_dirty = false;
    }

    /**
     * @param {number[]} index
     * @returns {number}
     */
    #calcIndex(...index){
        if(index.length !== this.shape.length){
            throw new Error(`Index mismatch: ${index.length} !== ${this.shape.length}`);
        }
        return index.reduce((a, _, i) => a + this.strides[i] * index[i], 0);
    }

    /**
     * Get Value
     * @param {number[]} index
     * @returns {Promise<number>}
     */
    async get(...index){
        await this.load();
        return this.get_without_load(...index);
    }

    /**
     * Get Value without Load
     * @param {number[]} index
     * @returns {number}
     */
    get_without_load(...index){
        if(index.length > this.shape.length){
            throw new Error(`Too many indices: ${index.length} > ${this.shape.length}`);
        }

        if(index.length === this.shape.length){
            return this.cpu.at(this.#calcIndex(...index));
        }

        throw new Error(`Not Implemented yet`);
    }

    /**
     * Set Value
     * @param {number} value
     * @param {number[]} index
     */
    set(value, ...index){
        if(index.length > this.shape.length){
            throw new Error(`Too many indices: ${index.length} > ${this.shape.length}`);
        }
        this.cpu_dirty = true;

        if(index.length === this.shape.length){
            this.cpu[this.#calcIndex(...index)] = value;
            return;
        }

        throw new Error(`Not Implemented yet`);
    }

    [Symbol.iterator](){
        if(this.gpu_dirty){
            console.warn(`There are unloaded data at GPU`);
        }
        return this.cpu[Symbol.iterator]();
    }
};


/**
 * Create GPU Instance
 * @param {GPUOptions?} options
 * @returns {GPUBackend}
 */
const createGPU = async (options) => {
    options ??= {};
    const { adapter, device } = options;

    const a = await navigator?.gpu?.requestAdapter(adapter)
    if(a === undefined){
        throw new Error(`No Available GPU Adapter`);
    }
    a.requestAdapterInfo().then(i => {
        console.log(`GPU Adapter
  vendor      : ${i.vendor}
  architecture: ${i.architecture}
  device      : ${i.device}
  description : ${i.description}`);

        console.log(["GPU Supported Features", ...a.features.keys()].join("\n  "));
        console.table(a.limits);
    });


    const d = await a.requestDevice(device);
    if(d === undefined){
        throw new Error(`No Available GPU`);
    }
    d.addEventListener("uncapturederror", ({ error }) => {
        const name = error.constructor.name;
        const message = error.message
        throw new Error(`Uncaptured WebGPU Error: ${name}: ${message}`);
    });

    return new GPUBackend(d);
};


export { createGPU };
