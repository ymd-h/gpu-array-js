import { createGPU } from "./gpu-array.js";
import { TEST, assertEqual, assertAlmostEqual } from "./test.js";


const gpu = await createGPU();
const N = 10;


const showArray = (array) => {
    console.log("[" + Array.from({length: N}, (_, row) => {
        return "[" + (
            Array.from(
                {length: N},
                (_, col) => array.get_without_load(row, col),
            )
        ).join() + "]";
    }).join("\n") + "]");
};

const arrayEach = (f, ...args) => {
    for(let col = 0; col < N; col++){
        for(let row = 0; row < N; row++){
            f(...args.map(a => a.get_without_load(row, col)));
        }
    }
};


const a = gpu.Array({ shape: [N, N] });
const b = gpu.Array({ shape: [N, N] });
const u32 = gpu.Array({ shape: [N, N], dtype: "u32" });

for(let col = 0; col < N; col++){
    for(let row = 0; row < N; row++){
        a.set(col * row, row, col);
        b.set(col + row, row, col);
        u32.set(row + col, row, col);
    }
}

const s = gpu.Array({ shape: [1,] });
s.set(0.2, 0);


TEST([
    ["a + b", async () => {
        const c = gpu.add(a, b);
        await c.load();
        arrayEach((ai, bi, ci) => assertAlmostEqual(ai + bi, ci), a, b, c);
    }],
    ["a - b", async () => {
        const d = gpu.sub(a, b);
        await d.load();
        arrayEach((ai, bi, di) => assertAlmostEqual(ai - bi, di), a, b, d);
    }],
    ["a * b", async () => {
        const e = gpu.mul(a, b);
        await e.load();
        arrayEach((ai, bi, ei) => assertAlmostEqual(ai * bi, ei), a, b, e);
    }],
    ["a / b", async () => {
        const f = gpu.div(a, b);
        await f.load();
        arrayEach((ai, bi, fi) => assertAlmostEqual(ai / bi, fi), a, b, f);
    }],
    ["log(a)", async () => {
        const g = gpu.log(a);
        await g.load();
        arrayEach((ai, gi) => ai < 0 || assertAlmostEqual(Math.log(ai), gi), a, b, g);
    }],
    ["pow(a, b)", async () => {
        const h = gpu.pow(a, b);
        await h.load();
        arrayEach((ai, bi, hi) => assertAlmostEqual(ai ** bi, hi), a, b, h);
    }],
    ["a + u32 (Type Conversion)", async () => {
        const a_u32 = gpu.add(a, u32);
        await a_u32.load();
        arrayEach((ai, ui, aui) => assertAlmostEqual(ai + ui, aui), a, u32, a_u32);
    }],
    ["a + s (Broadcast)", async () =>{
        const a_s = gpu.add(a, s);
        await a_s.load();
        arrayEach((ai, asi) => assertAlmostEqual(ai + s.get_without_load(0), asi),
                  a, a_s);
    }],
]);

