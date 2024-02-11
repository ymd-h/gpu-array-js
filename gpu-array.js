/** @module gpu-array */

import { vector_op, func1, func2 } from "./shader.js";


/**
 * @typedef {Object} AdaptorOptions
 * @property {"low-power" | "high-performance" | undefined} powerPreference
 *
 * @typedef {Object} DeviceOptions
 * @property {{label: string} | undefined} defaultQueue
 * @property {string?} label
 * @property {string[] | undefined} requiredFeatures
 * @property {Object.<string, *>} requiredLimits
 *
 * @typedef {Object} GPUOptions
 * @property {AdaptorOptions?} adaptor
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
        this.shader.set(code, shader);

        return shader;
    }

    /**
     * Create NDArray
     * @param {GPUDevice}
     * @param {ArrayOptions}
     * @returns {NDArray}
     */
    Array(options){
        this.assertLost();
        return new NDArray(this.device, options);
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

        out ??= this.Array({ shape: lhs.shape, dtype });
        const size = this.device.limits.maxComputeWorkgroupSizeX;

        const shader = this.createShader(
            vector_op(
                op, size,
                {binding: 0, type: lhs.dtype, conv: (dtype === lhs.dtype) ? "" : dtype},
                {binding: 1, type: rhs.dtype, conv: (dtype === rhs.dtype) ? "" : dtype},
                {binding: 2, type: out.dtype, conv: (dtype === out.dtype) ? "" : dtype},
            ),
        );

        this.execute(
            shader,
            [
                {array: lhs, mode: "read-only"},
                {array: rhs, mode: "read-only"},
                {array: out, mode: "write-only"},
            ],
            [Math.ceil(out.length / size)],
        );

        return out;
    }

    _func1(f, arg, out){
        out ??= this.Array({ shape: arg.shape, dtype: arg.dtype });
        const size = this.device.limits.maxComputeWorkgroupSizeX;

        const out_conv = (arg.dtype === out.dtype) ? "" : out.dtype;

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

        out ??= this.Array({ shape: arg0.shape, dtype });
        const size = this.device.limits.maxComputeWorkgroupSizeX;

        const conv = (v) => (v.dtype === dtype) ? "" : dtype;

        const shader = this.createShader(
            func2(
                f, size,
                [
                    { binding: 0, type: arg0.dtype, conv: conv(arg0) },
                    { binding: 1, type: arg1.dtype, conv: conv(arg1) },
                ],
                { binding: 2, type: out.dtype, conv: conv(out) },
            ),
        );

        this.execute(
            shader,
            [
                {array: arg0, mode: "read-only"},
                {array: arg1, mode: "read-only"},
                {array: out, mode: "write-only"},
            ],
            [Math.ceil(out.length / size)],
        );

        return out;
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
        shape ??= [1];
        if(typeof shape === "number"){
            shape = [shape];
        }
        if(shape.some(s => s <= 0)){
            throw new Error(`shape must be positive: [${shape.join()}]`);
        }
        this.shape = shape.map(s => s | 0);

        // strides
        if(strides !== undefined){
            throw new Error(`custom strides has not supported yet`);
        }
        strides ??= shape.reduce((a, si) => {
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
        this.length = shape.reduce((a, v) => a * v);

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
};


/**
 * Create GPU Instance
 * @param {GPUOptions?} options
 * @returns {GPUBackend}
 */
const createGPU = async (options) => {
    options ??= {};
    const { adaptor, device } = options;

    const d = await (
        await navigator?.gpu?.requestAdapter(adaptor)
    )?.requestDevice(device);

    if(d === undefined){
        throw new Error(`No Available GPU`);
    }

    return new GPUBackend(d);
};


export { createGPU };
