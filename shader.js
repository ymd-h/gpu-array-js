/** @module shader */

const f16 = (...arrays) => {
    return arrays.some(a => a.type === "f16") ? "enable f16;" : "";
};

const vector_op = (op, size, lhs, rhs, out) => `
${f16(lhs, rhs, out)}
@group(0) @binding(${lhs.binding})
var<storage, read> lhs: array<${lhs.type}>;

@group(0) @binding(${rhs.binding})
var<storage, read> rhs: array<${rhs.type}>;

@group(0) @binding(${out.binding})
var<storage, read_write> out: array<${out.type}>;

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    out[id.x] = ${out.conv}(${lhs.conv}(lhs[id.x]) ${op} ${rhs.conv}(rhs[id.x]));
}
`;

const vector_op_indirect = (
    op, size,
    lhs, rhs, out,
    lhs_strides, rhs_strides, out_strides,
) => `
${f16(lhs, rhs, out)}
@group(0) @binding(${lhs.binding})
var<storage, read> lhs: array<${lhs.type}>;

@group(0) @binding(${rhs.binding})
var<storage, read> rhs: array<${rhs.type}>;

@group(0) @binding(${out.binding})
var<storage, read_write> out: array<${out.type}>;

@group(0) @binding(${lhs_strides.binding})
var<storage, read> lhs_strides: array<u32>;

@group(0) @binding(${rhs_strides.binding})
var<storage, read> rhs_strides: array<u32>;

@group(0) @binding(${out_strides.binding})
var<storage, read> out_strides: array<u32>;

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    var O: u32 = id.x;
    var L: u32 = 0;
    var R: u32 = 0;
    for(var s: u32 = arrayLength(&out_strides) -1; s > 0; s--){
        let iN: u32 = O % out_strides[s-1];
        var i: u32 = iN / out_strides[s];
        L += i * lhs_strides[s];
        R += i * rhs_strides[s];
        O -= iN;
    }
    var i: u32 = O / out_strides[0];
    L += i * lhs_strides[0];
    R += i * rhs_strides[0];

    out[id.x] = ${out.conv}(${lhs.conv}(lhs[L]) ${op} ${rhs.conv}(rhs[R]));
}
`;


const func1 = (f, size, arg, out) => `
${f16(arg, out)}
@group(0) @binding(${arg.binding})
var<storage, read> arg: array<${arg.type}>;

@group(0) @binding(${out.binding})
var<storage, read_write> out: array<${out.type}>;

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    out[id.x] = ${out.conv}(${f}(${arg.conv}(arg[id.x])));
}
`;

const func2 = (f, size, args, out) => `
${f16(...args, size)}
@group(0) @binding(${args[0].binding})
var<storage, read> arg0: array<${args[0].type}>;

@group(0) @binding(${args[1].binding})
var<storage, read> arg1: array<${args[1].type}>;

@group(0) @binding(${out.binding})
var<storage, read_write> out: array<${out.type}>;

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    out[id.x] = ${out.conv}(${f}(${args[0].conv}(arg0[id.x]), ${args[1].conv}(arg1[id.x])));
}
`;

const func2_indirect = (f, size, args, out, args_strides, out_strides) => `
${f16(...args, out)}
@group(0) @binding(${args[0].binding})
var<storage, read> arg0: array<${args[0].type}>;

@group(0) @binding(${args[1].binding})
var<storage, read> arg1: array<${args[1].type}>;

@group(0) @binding(${out.binding})
var<storage, read_write> out: array<${out.type}>;

@group(0) @binding(${args_strides[0].binding})
var<storage, read> arg0_strides: array<u32>;

@group(0) @binding(${args_strides[1].binding})
var<storage, read> arg1_strides: array<u32>;

@group(0) @binding(${out_strides.binding})
var<storage, read> out_strides: array<u32>;

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    var O: u32 = id.x;
    var I0: u32 = 0;
    var I1: u32 = 0;
    for(var s: u32 = arrayLength(&out_strides) -1; s > 0; s--){
        let iN: u32 = O % out_strides[s-1];
        var i: u32 = iN / out_strides[s];
        I0 += i * arg0_strides[s];
        I1 += i * arg1_strides[s];
        O -= i;
    }
    var i: u32 = O / out_strides[0];
    I0 += i * arg0_strides[0];
    I1 += i * arg1_strides[0];

    out[id.x] = ${out.conv}(${f}(${args[0].conv}(arg0[I0]), ${args[1].conv}(arg1[I1])));
}
`;


const reduce_op = (op, size, arg, out) => `
${f16(arg, out)}
@group(0) @binding(${arg.binding})
var<storage, read> arg: array<${arg.type}>;

@group(0) @binding(${out.binding})
var<storage, read_write> out: array<${out.type}>;

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    let N: u32 = arrayLength(&arg);
    let size: u32 = ${size};
    var v = arg[id.x];
    for(var i = id.x + size; i < N; i += size){
        v = v ${op} arg[i];
    }

    out[id.x] = ${out.conv}(v);
}
`;


const reduce_func = (f, size, arg, out) => `
${f16(arg, out)}
@group(0) @binding(${arg.binding})
var<storage, read> arg: array<${arg.type}>;

@group(0) @binding(${out.binding})
var<storage, read_write> out: array<${out.type}>;

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    let N: u32 = arrayLength(&arg);
    let size: u32 = ${size};
    var v = ${arg.conv}(arg[id.x]);
    for(var i = id.x + size; i < N; i += size){
        v = ${f}(v, ${arg.conv}(arg[i]));
    }

    out[id.x] = ${out.conv}(v);
}
`;


const _xoshiro128pp_out = (out) => (out === undefined) ?
      "" :
      `out[i] = ${(out.type === 'f32') ? 'toFloat' : ''}(rotl(s[0] + s[3], 7) + s[0]);`;


const _xoshiro128pp_binding = (out) => (out === undefined) ?
      "" :
      `@group(0) @binding(${out.binding}) var<storage, read_write> out: array<${out.type}>;`;

const _xoshiro128pp_next = (state, out) => `
${_xoshiro128pp_binding(out)}


fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32 - k));
}

fn toFloat(x: u32) -> f32 {
    return bitcast<f32>((x >> 9) | 0x3f800000) - 1.0;
}

fn next(i: u32){
    var s: vec4<u32> = state[i];
    ${_xoshiro128pp_out(out)}

    let t: u32 = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 11);

    state[i] = s;
}
`;

const xoshiro128pp = (size, state, out) => `
@group(0) @binding(${state.binding})
var<storage, read_write> state: array<vec4<u32>>;

${_xoshiro128pp_next(state, out)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if((id.x >= arrayLength(&state)) ${(out !== undefined) ? "|| (id.x >= arrayLength(&out))" : ""}){ return; }

    next(id.x);
}
`;


const xoshiro128pp_init = (state) => `
@group(0) @binding(${state.binding})
var<storage, read_write> state: array<vec4<u32>>;

${_xoshiro128pp_next(state)}

const JUMP: vec4<u32> = vec4(0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b);

fn jump(i: u32){
    var s: vec4<u32> = vec4(0, 0, 0, 0);
    for(var i: u32 = 0; i < 4; i++){
        for(var b: u32 = 0; b < 32; b++){
            if((JUMP[i] & (1u << b)) != 0){
                s ^= state[i];
            }
            next(i);
        }
    }

    state[i] = s;
}

@compute @workgroup_size(1)
fn main(){
    let n: u32 = arrayLength(&state);
    for(var i: u32 = 1; i < n; i++){
        state[i] = state[i-1];
        jump(i);
    }
}
`;


export {
    vector_op, vector_op_indirect,
    func1,
    func2, func2_indirect,
    reduce_op, reduce_func,
    xoshiro128pp, xoshiro128pp_init,
};
