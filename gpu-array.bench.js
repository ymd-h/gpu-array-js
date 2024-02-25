import { createGPU } from "./gpu-array.js";
import { BENCH } from "./bench.js";


const gpu = await createGPU();

const N = 1000000;
const a = Array.from({length: N}, (_, i) => i);
const b = Array.from({length: N}, (_, i) => i * 2 + 3);


await BENCH(`Add: N = ${N}`, [
    [`js`, () => {
        return Array.from({length: N}, (_, i) => a[i] + b[i]);
    }],
    [`gpu (set bulk)`, async () => {
        const A = gpu.Array({shape: N});
        A.set(a);

        const B = gpu.Array({shape: N});
        B.set(b);

        const C = gpu.add(A, B);
        await C.load();

        return C;
    }],
    [`gpu (set one by one)`, async () => {
        const A = gpu.Array({shape: N});
        a.forEach((ai, i) => A.set(ai, i));

        const B = gpu.Array({shape: N});
        b.forEach((bi, i) => B.set(bi, i));

        const C = gpu.add(A, B);
        await C.load();

        return C;
    }],
]);


await BENCH(`sqrt: N = ${N}`, [
    [`js`, () => {
        return Array.from({length: N}, (_, i) => Math.sqrt(a[i]));
    }],
    [`gpu (set bulk)`, async () => {
        const A = gpu.Array({shape: N});
        A.set(a);

        const C = gpu.sqrt(A);
        await C.load();

        return C;
    }],
    [`gpu (set one by one)`, async () => {
        const A = gpu.Array({shape: N});
        a.forEach((ai, i) => A.set(ai, i));

        const C = gpu.sqrt(A);
        await C.load();

        return C;
    }],
]);
