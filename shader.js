/** @module shader */

const f16 = (...arrays) => {
    return arrays.some(a => a.type === "f16") ? "enable f16;" : "";
};

const binding = (name, arg, write = false) => (arg.scalar !== undefined) ?
      `override ${name}: ${arg.type};` :
      `@group(0) @binding(${arg.binding})
var<storage, ${write ? "read_write" : "read"}> ${name}: array<${arg.type ?? "u32"}>;`;


const v = (name, arg, idx) => (arg.scalar !== undefined) ? name : `${name}[${idx}]`;
const s = (stmt, arg) => (arg.scalar !== undefined) ? "" : stmt;

const vector_op = (op, size, lhs, rhs, out) => `
${f16(lhs, rhs, out)}

${binding("lhs", lhs)}

${binding("rhs", rhs)}

${binding("out", out, true)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    let L = ${lhs.conv ?? ""}(${v("lhs", lhs, "id.x")});
    let R = ${rhs.conv ?? ""}(${v("rhs", rhs, "id.x")});
    out[id.x] = ${out.conv}(L ${op} R);
}
`;

const vector_op_indirect = (
    op, size,
    lhs, rhs, out,
    lhs_strides, rhs_strides, out_strides,
) => `
${f16(lhs, rhs, out)}

${binding("lhs", lhs)}

${binding("rhs", rhs)}

${binding("out", out, true)}

${s(binding("lhs_strides", lhs_strides), lhs_strides)}

${s(binding("rhs_strides", rhs_strides), rhs_strides)}

${binding("out_strides", out_strides)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    var O: u32 = id.x;
    ${s("var L: u32 = 0;", lhs)}
    ${s("var R: u32 = 0;", rhs)}
    for(var s: u32 = arrayLength(&out_strides) -1; s > 0; s--){
        let iN: u32 = O % out_strides[s-1];
        var i: u32 = iN / out_strides[s];
        ${s("L += i * lhs_strides[s];", lhs)}
        ${s("R += i * rhs_strides[s];", rhs)}
        O -= iN;
    }
    var i: u32 = O / out_strides[0];
    ${s("L += i * lhs_strides[0];", lhs)}
    ${s("R += i * rhs_strides[0];", rhs)}

    let LHS = ${lhs.conv ?? ""}(${v("lhs", lhs, "L")});
    let RHS = ${rhs.conv ?? ""}(${v("rhs", rhs, "R")});

    out[id.x] = ${out.conv}(LHS ${op} RHS);
}
`;

const func1 = (f, size, arg, out) => `
${f16(arg, out)}

${binding("arg", arg)}

${binding("out", out, true)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    out[id.x] = ${out.conv}(${f}(${arg.conv}(${v("arg", arg, "id.x")})));
}
`;

const func2 = (f, size, args, out) => `
${f16(...args, size)}

${binding("arg0", args[0])}

${binding("arg1", args[1])}

${binding("out", out, true)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    let A0 = ${args[0].conv}(${v("arg0", args[0], "id.x")});
    let A1 = ${args[1].conv}(${v("arg1", args[1], "id.x")});
    out[id.x] = ${out.conv}(${f}(A0, A1));
}
`;

const func2_indirect = (f, size, args, out, args_strides, out_strides) => `
${f16(...args, out)}

${binding("arg0", args[0])}

${binding("arg1", args[1])}

${binding("out", out, true)}

${binding("arg0_strides", args_strides[0])}

${binding("arg1_strides", args_strides[1])}

${binding("out_strides", out_strides)}

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

${binding("arg", arg)}

${binding("out", out, true)}

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

${binding("arg", arg)}

${binding("out", out, true)}

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
      `${binding("out", out, true)}`;

const _xoshiro128pp_next = (state, out) => `
${binding("state", { ...state, type: "vec4<u32>"}, true)}

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
${_xoshiro128pp_next(state, out)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if((id.x >= arrayLength(&state)) ${(out !== undefined) ? "|| (id.x >= arrayLength(&out))" : ""}){ return; }

    next(id.x);
}
`;


const xoshiro128pp_init = (state) => `
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

const box_muller = (size, u, v, r1, r2) => `
${binding("u", u)}

${binding("v", v)}

${binding("r1", r1, true)}

${binding("r2", r2, true)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&r1)){ return; }

    let theta = ${2 * Math.PI} * u[id.x];
    let R = sqrt(-2 * log(1 - v[id.x]));

    r1[id.x] = R * cos(theta);
    r2[id.x] = R * sin(theta);
}
`;


const flat_index = (size, strides, index, out) => `
${binding("strides", strides)}

${binding("index", index)}

${binding("out", out, true)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    let n: u32 = arrayLength(&strides);
    var I: u32 = 0;
    for(var i = 0; i < n; i++){
        I += index[id.x * n + i] * strides[i];
    }
    out[id.x] = I;
}
`;

const gather = (size, from, fromIndex, to, toIndex) => `
${f16(from, to)}

${binding("from", from)}

${binding("fromIndex", fromIndex)}

${binding("to", to)}

${binding("toIndex", toIndex)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if((id.x >= arrayLength(&fromIndex)) || (id.x >= arrayLength(&toIndex))){ return; }

    to[toIndex[id.x]] = from[fromIndex[id.x]];
}
`;


const where = (size, cond, True, False, out) => `
${f16(True, False, out)}

${binding("cond", cond)}

${binding("True", True)}

${binding("False", False)}

${binding("out", out, true)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    let F = ${False.conv ?? ""}(${v("False", False, "id.x")});
    let T = ${ True.conv ?? ""}(${v( "True",  True, "id.x")});
    let C = bool(${v("cond", cond, "id.x")});
    out[id.x] = ${out.conv ?? ""}(select(F, T, C));
}
`;


const where_indirect = (
    size,
    cond, True, False, out,
    cond_strides, True_strides, False_strides, out_strides) => `
${f16(True, False, out)}

${binding("cond", cond)}

${binding("True", True)}

${binding("False", False)}

${binding("out", out, true)}

${binding("cond_strides", cond_strides)}

${binding("True_strides", True_strides)}

${binding("False_strides", False_strides)}

${binding("out_strides", out_strides)}

@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) id: vec3<u32>){
    if(id.x >= arrayLength(&out)){ return; }

    var O: u32 = id.x;
    var C: u32 = 0;
    var T: u32 = 0;
    var F: u32 = 0;
    for(var s: u32 = arrayLength(&out_strides) -1; s > 0; s--){
        let iN: u32 = O % out_strides[s-1];
        var i: u32 = iN / out_strides[s];
        C += i * cond_strides[s];
        T += i * True_strides[s];
        F += i * False_strides[s];
        O -= i;
    }
    var i: u32 = O / out_strides[0];
    C += i * cond_strides[0];
    T += i * True_strides[0];
    F += i * False_strides[0];

    out[id.x] = ${out.conv ?? ""}(select(${False.conv ?? ""}(False[F]), ${True.conv ?? ""}(True[T]), bool(cond[C])));
}
`;


export {
    vector_op, vector_op_indirect,
    func1,
    func2, func2_indirect,
    reduce_op, reduce_func,
    xoshiro128pp, xoshiro128pp_init,
    box_muller,
    flat_index, gather,
    where, where_indirect,
};
