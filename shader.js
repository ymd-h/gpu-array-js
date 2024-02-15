/** @module shader */

const vector_op = (op, size, lhs, rhs, out) => `
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


export {
    vector_op, vector_op_indirect,
    func1,
    func2, func2_indirect,
};
