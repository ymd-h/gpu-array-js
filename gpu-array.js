/** @module gpu-array */

import { Float16Array } from "https://cdn.jsdelivr.net/npm/@petamoriken/float16/+esm";

import {
    vector_op, vector_op_indirect,
    func1,
    func2, func2_indirect,
    reduce_op, reduce_func,
    xoshiro128pp, xoshiro128pp_init,
    box_muller,
    where, where_indirect,
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
 *
 * @typedef {Object} PRNGOptions
 * @property {number | bigint | undefined} seed
 * @property {number?} size
 */


/**
 * @param {number[][]} shapes
 * @returns {bool}
 */
const equalShapes = (...shapes) => {
    shapes = shapes.filter(s => s !== undefined);

    if(shapes.length <= 1){
        return true;
    }

    const first = shapes.shift();

    if(shapes.some(s => s.length !== first.length)){
        return false;
    }

    if(shapes.some(s => s.some((si, i) => si !== first[i]))){
        return false;
    }

    return true;
};


/**
 * @param {string?} t1
 * @param {string?} t2
 * @returns {string}
 */
const promoteType = (t1, t2) => {
    if(t1 === t2){
        return t1 ?? "f32";
    }

    if((t1 === "f32") || (t2 === "f32")){
        return "f32";
    }

    if((t1 === "f16") || (t2 === "f16")){
        return "f16";
    }

    if((t1 !== undefined) && (t2 !== undefined)){
        throw new Error(`Incompatible Types: ${t1}, ${t2}`);
    }

    return t1 ?? t2;
}


/**
 * @param {number[][]} shapes
 * @returns {number[]}
 */
const broadcastShapes = (...shapes) => {
    shapes = shapes.map(s => s ?? [1]);
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

        /** @type {Map<GPUShaderModule, GPUComputePipeliine>} */
        this.pipe = new Map();

        // Vector Operand
        const vop = [
            ["add", "+"],
            ["sub", "-"],
            ["mul", "*"],
            ["div", "/"],
        ];
        for(const [name, op] of vop){
            this[name] = (lhs, rhs, out) => this._vector_op(op, lhs, rhs, out);
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
            this[f] = (arg, out) => this._func1(f, arg, out);
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
                    console.error(code);
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
     * @returns {NDArray}
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

    /**
     * @param {number[] | TypedArray} value
     * @param {ArrayOptions?} options
     * @returns {NDArray}
     */
    asarray(value, options){
        const o = { shape: value.length };
        if(value instanceof Uint32Array){
            o.dtype = "u32";
        } else if(value instanceof Int32Array){
            o.dtype = "i32";
        } else if(value instanceof Float16Array){
            o.dtype = "f16";
        } else if(value instanceof Float32Array){
            o.dtype = "f32";
        }

        const a = this.Array({ ...o, ...options });
        a.set(value);

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

    _destroyOnDone(...arrays){
        this.device.queue.onSubmittedWorkDone().then(() => {
            arrays.forEach(a => a?.gpu.destroy());
        });
    }

    /**
     * Execute GPU Computation
     * @param {GPUShaderModule} shader
     * @param {ArraySpec[]} specs
     * @param {number[]} dispatch
     * @param {Object.<string, *>?} constants
     */
    execute(shader, specs, dispatch, constants){
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
                    constants,
                };
            }),
        });

        let pipeline = null;
        if((constants === undefined) && this.pipe.has(shader)){
            pipeline = this.pipe.get(shader);
        } else {
            pipeline = this.device.createComputePipeline({
                layout: pipelineLayout,
                compute: {
                    module: shader,
                    entryPoint: "main",
                    constants,
                },
            });
            if(constants === undefined){
                this.pipe.set(shader, pipeline);
            }
        }


        const cmd = this.device.createCommandEncoder();
        const pass = cmd.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...dispatch);
        pass.end();
        this.device.queue.submit([cmd.finish()]);
    }

    /**
     * @returns {number}
     */
    get sizeX(){
        return 64;
    }

    _vector_op(op, lhs, rhs, out){
        const dtype = promoteType(lhs.dtype, rhs.dtype);

        out ??= this.Array({ shape: broadcastShapes(lhs.shape, rhs.shape), dtype });
        const size = this.sizeX;

        if(out.custom_strides){
            throw new Error(`Custom Strides for out is not supported`);
        }

        const lhs_array = lhs instanceof NDArray;
        const rhs_array = rhs instanceof NDArray;

        let b = 0;
        const shader_args = [
            op, size,
            lhs_array ? {
                binding: b++,
                type: lhs.dtype,
                conv: (dtype === lhs.dtype) ? "" : dtype,
            } : {
                scalar: true,
                type: dtype,
            },
            rhs_array ? {
                binding: b++,
                type: rhs.dtype,
                conv: (dtype === rhs.dtype) ? "" : dtype,
            } : {
                scalar: true,
                type: dtype,
            },
            {
                binding: b++,
                type: out.dtype,
                conv: (dtype === out.dtype) ? "" : out.dtype,
            },
        ];

        const execute_buffers = [];
        if(lhs_array){
            execute_buffers.push({array: lhs, mode: "read-only"});
        }
        if(rhs_array){
            execute_buffers.push({array: rhs, mode: "read-only"});
        }
        execute_buffers.push({array: out, mode: "write-only"});


        const constants = (lhs_array && rhs_array) ? undefined : {};
        if(!lhs_array){
            constants.lhs = lhs;
        }
        if(!rhs_array){
            constants.rhs = rhs;
        }

        let lhs_strides = null;
        let rhs_strides = null;
        let out_strides = null;

        const use_strides = (lhs.custom_strides ||
                             rhs.custom_strides ||
                             !equalShapes(lhs.shape, rhs.shape, out.shape));
        if(use_strides){
            if(lhs_array){
                lhs_strides = this.#stridesBuffer(broadcastStrides(lhs, out.shape));
                shader_args.push({binding: b++});
                execute_buffers.push({array: lhs_strides, mode: "read-only"});
            } else {
                shader_args.push({scalar: true});
            }

            if(rhs_array){
                rhs_strides = this.#stridesBuffer(broadcastStrides(rhs, out.shape));
                shader_args.push({binding: b++});
                execute_buffers.push({array: rhs_strides, mode: "read-only"});
            } else {
                shader_args.push({scalar: true});
            }

            out_strides = this.#stridesBuffer(out.strides);
            shader_args.push(
                {binding: b++},
            );
            execute_buffers.push(
                {array: out_strides, mode: "read-only"},
            );
        }

        const shader = this.createShader(
            use_strides ?
                vector_op_indirect(...shader_args):
                vector_op(...shader_args),
        );

        this.execute(
            shader,
            execute_buffers,
            [Math.ceil(out.length / size)],
            constants,
        );
        this._destroyOnDone(lhs_strides, rhs_strides, out_strides);
        return out;
    }

    _func1(f, arg, out){
        out ??= this.Array({ shape: arg.shape, dtype: arg.dtype });
        const size = this.sizeX;

        if(out.custom_strides){
            throw new Error(`Custom Strides for out is not supported`);
        }

        const out_conv = (arg.dtype === out.dtype) ? "" : out.dtype;

        if(arg.custom_strides ||
           !equalShapes(arg.shape, out.shape)){
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
        const dtype = promoteType(arg0.dtype, arg1.dtype);

        out ??= this.Array({ shape: broadcastShapes(arg0.shape, arg1.shape), dtype });
        const size = this.sizeX;

        if(out.custom_strides){
            throw new Error(`Custom Strides for out is not supported`);
        }

        const conv = (v) => (v.dtype === dtype) ? "" : dtype;
        const out_conv = (out.dtype === dtype) ? "" : out.dtype;

        const use_strides = (arg0.custom_strides ||
                             arg1.custom_strides ||
                             !equalShapes(arg0.shape, arg1.shape, out.shape));

        const shader_args = [
            f, size,
            [
                { binding: 0, type: arg0.dtype, conv: conv(arg0) },
                { binding: 1, type: arg1.dtype, conv: conv(arg1) },
            ],
            { binding: 2, type: out.dtype, conv: out_conv },
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
        this._destroyOnDone(arg0_strides, arg1_strides, out_strides);
        return out;
    }

    _reduce_op(op, arg){
        if(arg.custom_strides){
            throw new Error(`Reduce Op hasn't supported custom strides yet`);
        }

        while(true){
            const length = (arg.length > 64) ?
                  Math.min(1 << (Math.floor(Math.log2(arg.length)) -1), this.sizeX):
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
            this._destroyOnDone(out);
        }
    }

    _reduce_func(f, arg){
        if(arg.custom_strides){
            throw new Error(`Reduce Func hasn't supported custom strides yet`);
        }

        while(true){
            const length = (arg.length > 64) ?
                  Math.min(1 << (Math.floor(Math.log2(arg.length)) -1), this.sizeX):
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
            this._destroyOnDone(out);
        }
    }

    /**
     * @param {NDArray} cond
     * @param {NDArray} True
     * @param {NDArray} False
     * @param {NDArray?} out
     * @returns {NDArray}
     */
    where(cond, True, False, out){
        out ??= this.Array({
            shape: broadcastShapes(cond.shape, True.shape, False.shape),
            dtype: promoteType(True.dtype, False.dtype),
        });

        if(out.custom_strides){
            throw new Error(`Custom Strides of out is not supported.`);
        }

        const use_strides = (cond.custom_strides ||
                             True.custom_strides ||
                             False.custom_strides ||
                             out.custom_strides ||
                             !equalShapes(cond.shape,
                                          True.shape,
                                          False.shape,
                                          out.shape));

        const shader_args = [
            this.sizeX,
            {binding: 0, type: cond.dtype},
            {binding: 1, type: True.dtype},
            {binding: 2, type: False.dtype},
            {binding: 3, type: out.dtype},
        ];

        const execute_buffers = [
            {array: cond, mode: "read-only"},
            {array: True, mode: "read-only"},
            {array: False, mode: "read-only"},
            {array: out, mode: "write-only"},
        ];

        let cond_strides = null;
        let True_strides = null;
        let False_strides = null;
        let out_strides = null;

        if(use_strides){
            shader_args.push(
                {binding: 4}, // cond_strides
                {binding: 5}, // True_strides
                {binding: 6}, // False_strides
                {binding: 7}, // out_strides
            );

            cond_strides = this.#stridesBuffer(broadcastStrides(cond, out.shape));
            True_strides = this.#stridesBuffer(broadcastStrides(True, out.shape));
            False_strides = this.#stridesBuffer(broadcastStrides(False, out.shape));
            out_strides = this.#stridesBuffer(out.strides);

            execute_buffers.push(
                {array: cond_strides, mode: "read-only"},
                {array: True_strides, mode: "read-only"},
                {array: False_strides, mode: "read-only"},
                {array: out_strides, mode: "read-only"},
            );
        }

        const shader = this.createShader(
            use_strides ?
                where_indirect(...shader_args) :
                where(...shader_args)
        );

        this.execute(shader, execute_buffers, [Math.ceil(out.length / this.sizeX)]);
        this._destroyOnDone(cond_strides, True_strides, False_strides, out_strides);

        return out;
    }

    /**
     * @param {PRNGOptions?} options
     * @returns {Xoshiro128pp}
     */
    Xoshiro128pp(options){
        return new Xoshiro128pp(this, options);
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
            this.cpu = new Float16Array(this.length);
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

        const _size = this.length * this.itemsize;
        if(_size % 4){
            throw new Error(`Data Size must be multiple of 4 bytes, but ${this.itemsize} bytes x ${this.length} items = ${_size} bytes`);
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

        // Float16Array ponyfill cannot be passed directly.
        this.device.queue.writeBuffer(this.gpu, 0, this.cpu.buffer);
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
     * @param {number | number[] | TypedArray} value
     * @param {number[]} index
     */
    set(value, ...index){
        if(index.length === 0){
            if(value.length === this.length){
                this.cpu.set(value);
                this.cpu_dirty = true;
                return;
            }
            throw new Error(`Incompatible length: ${this.length} !== ${value.length}`);
        }

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


class PRNG {
    #rng

    /**
     * @param {"f32" | "f16"} dtype
     * @returns {NDArray}
     */
    normal(dtype){
        dtype ??= "f32";

        if(this.#rng?.dtype === dtype){
            const rng = this.#rng;
            this.#rng = null;
            return rng;
        }

        // 0 <= u, v < 1
        const u = this.next(dtype);
        const v = this.next(dtype);

        const b = this.backend;

        const r1 = b.Array({ shape: u.shape, dtype });
        const r2 = b.Array({ shape: u.shape, dtype });

        const shader = b.createShader(
            box_muller(
                b.sizeX,
                {binding: 0, type: u.dtype},
                {binding: 1, type: v.dtype},
                {binding: 2, type: r1.dtype},
                {binding: 3, type: r2.dtype},
            ),
        );

        b.execute(
            shader,
            [
                {array: u, mode: "read-only"},
                {array: v, mode: "read-only"},
                {array: r1, mode: "write-only"},
                {array: r2, mode: "write-only"},
            ],
            [Math.ceil(u.length / b.sizeX)],
        );

        this.#rng = r2;

        b._destroyOnDone(u, v);
        return r1;
    }
};

class Xoshiro128pp extends PRNG {
    /**
     * @param {GPUBackend} backend
     * @param {PRNGOptions?} options
     */
    constructor(backend, options){
        super();

        let { seed, size } = options ?? {};

        /** @type {GPUBackend} */
        this.backend = backend;

        /** @type {number} */
        this.size = size ?? 64;

        seed ??= crypto.getRandomValues(new BigUint64Array(1))[0];

        // SplitMix64
        seed = BigInt(seed) & ((2n ** 64n) -1n);
        const state = Array.from({length: 4}, () => {
            let z = (seed += 0x9e3779b97f4a7c15n);
            z = (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n;
            z = (z ^ (z >> 27n)) * 0x94d049bb133111ebn;
            return Number((z ^ (z >> 31n)) & ((2n ** 32n) -1n));
        });

        /** @type {NDArray} */
        this.state = this.backend.Array({ shape: [this.size, 4], dtype: "u32" });
        this.state.cpu.set(state);
        this.state.cpu_dirty = true;

        const shader = this.backend.createShader(xoshiro128pp_init({binding: 0}));
        this.backend.execute(shader, [{array: this.state, mode: "read-write"}], [1]);
    }

    /**
     * @param {"u32" | "f32"} dtype
     * @returns {NDArray}
     */
    next(dtype){
        dtype ??= "u32";
        if(!["u32", "f32"].includes(dtype)){
            throw new Error(`Incompatible dtype: ${dtype}`);
        }

        const out = this.backend.Array({ shape: this.size, dtype });

        const size = Math.min(this.size, this.backend.sizeX);
        const shader = this.backend.createShader(xoshiro128pp(
            size,
            {binding: 0},
            {binding: 1, type: out.dtype},
        ));

        this.backend.execute(
            shader,
            [
                {array: this.state, mode: "read-write"},
                {array: out, mode: "write-only"},
            ],
            [Math.ceil(this.size / size)],
        );

        return out;
    }
};


/**
 * Create GPU Instance
 * @param {GPUOptions?} options
 * @returns {GPUBackend}
 */
const createGPU = async (options) => {
    options ??= {};
    let { adapter, device } = options;

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

    device ??= {};
    const f16 = "shader-f16";
    if(a.features.has(f16) && !device.requiredFeatures?.includes(f16)){
        console.log(`${f16} is added.`);
        device["requiredFeatures"] = [f16, ...device["requiredFeatures"] ?? []];
    }

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
