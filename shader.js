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

    out[id.x] = ${out.conv}(${f}(${args[0].conv}(arg0[id.x]),
                                 ${args[1].conv}(arg1[id.x])));
}
`;


export { vector_op, func1, func2 };
