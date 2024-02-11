import { createGPU } from "./gpu-array.js";


const gpu = await createGPU();

const N = 10;

const assertEach = async (f, msg, ...args) => {
    for(let col = 0; col < N; col++){
        for(let row = 0; row < N; row++){
            const v = await Promise.all(args.map(arg => arg.get(row, col)));
            if(!f(...args)){
                throw new Error(msg(...arg));
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


