import { createGPU } from "./gpu-array.js";


const gpu = await createGPU();

const N = 10;

const assertEach = async (f, msg, ...args) => {
    for(let col = 0; col < N; col++){
        for(let row = 0; row < N; row++){
            const v = await Promise.all(args.map(arg => arg.get(row, col)));
            if(!f(...v)){
                throw new Error(msg(...v));
            }
        }
    }
}

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


const a = gpu.Array({ shape: [N, N] });
const b = gpu.Array({ shape: [N, N] });

for(let col = 0; col < N; col++){
    for(let row = 0; row < N; row++){
        a.set(col * row, row, col);
        b.set(col + row, row, col);
    }
}
showArray(a);
showArray(b);

const c = gpu.add(a, b);
const d = gpu.sub(a, b);
const e = gpu.mul(c, a);
const f = gpu.div(e, a);


await c.load();
await assertEach((ai, bi, ci) => (ai + bi) !== ci,
                 (ai, bi, ci) => `${ai} + {bi} !== ${ci}`,
                 a, b, c);
console.log(`OK: a + b === c`);
showArray(c);

await d.load();
await assertEach((ai, bi, di) => (ai - bi) !== di,
                 (ai, bi, di) => `${ai} - ${bi} !== ${di}`,
                 a, b, d);
console.log(`OK: a - b === d`);

await e.load();
await assertEach((ci, ai, ei) => (ci * ai) !== ei,
                 (ci, ai, ei) => `${ci} * ${ai} !== ${ei}`,
                 c, a, e);
console.log(`OK: c * a === e`);

await f.load();
await assertEach((ei, ai, fi) => (ei / ai) !== fi,
                 (ei, ai, fi) => `${ei} / ${ai} !== ${fi}`,
                 e, a, f);
console.log(`OK: e / a === f`);


const g = gpu.log(a);
const h = gpu.pow(a, b);


await g.load();
await assertEach((ai, gi) => Math.log(ai) !== gi,
                 (ai, gi) => `$log({ai}) !== ${gi}`,
                 a, g);
console.log(`OK: log(a) === g`);
showArray(g);


await h.load();
await assertEach((ai, bi, hi) => (ai ** bi) !== hi,
                 (ai, bi, hi) => `pow(${ai}, ${bi}) !== ${hi}`,
                 a, b, h);
console.log(`OK: pow(a, b) === h`);
showArray(h);


const u32 = gpu.Array({ shape: [N, N], dtype: "u32" });
for(let row = 0; row < N; row++){
    for(let col = 0; col < N; col++){
        u32.set(row + col, row, col);
    }
}
const a_u32 = gpu.add(a, u32);

await a_u32.load();
await assertEach((ai, ui, aui) => (ai + ui) !== aui,
                 (ai, ui, aui) => `${ai} + ${ui} !== ${aui}`,
                 a, u32, a_u32);
console.log(`OK: a + u32 === a_u32 (type promotion)`);
