import { createGPU } from "./gpu-array.js";
import { TEST, assertEqual, assertAlmostEqual } from "./test.js";


const gpu = await createGPU();

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
            try {
                f(...args.map(a => a.get_without_load(row, col)));
            } catch(e){
                throw new Error(`(${row}, ${col}): ${e.message}`);
            }
        }
    }
};

TEST("Array Creation", [
    ["ones", async () => {
        const a = gpu.ones({ shape: [2, 2] });
        assertEqual(a.length, 4);
        assertAlmostEqual(await a.get(0, 0), 1);
        assertAlmostEqual(await a.get(0, 1), 1);
        assertAlmostEqual(await a.get(1, 0), 1);
        assertAlmostEqual(await a.get(1, 1), 1);
    }],
    ["ones: u32", async () => {
        const a = gpu.ones({ dtype: "u32" });
        assertAlmostEqual(a, [1]);
    }],
    ["full", async () => {
        const a = gpu.full(2);
        assertEqual(a.length, 1);
        assertAlmostEqual(await a.get(0), 2);
    }],
    ["arange", async () => {
        const a = gpu.arange({ stop: 3 });
        assertEqual(a.length, 3);
        assertAlmostEqual(await a.get(0), 0);
        assertAlmostEqual(await a.get(1), 1);
        assertAlmostEqual(await a.get(2), 2);
    }],
    ["arange with start", async () => {
        const a = gpu.arange({ start: 2, stop: 5 });
        assertAlmostEqual(a, [2, 3, 4]);
    }],
    ["arange with step", async () => {
        const a = gpu.arange({ stop: 3, step: 0.5 }, { dtype: "f32" });
        assertEqual(a.length, 6);
        assertAlmostEqual(a, [0, 0.5, 1.0, 1.5, 2, 2.5]);
    }],
    ["arange with negative step", async () => {
        const a = gpu.arange({ stop: -2, step: -1 });
        assertAlmostEqual(a, [0, -1]);
    }],
]);


TEST("Operator", [
    ["a + b", async () => {
        const a = gpu.full(2, { shape: [2, 2] });
        const b = gpu.arange({ stop: 4 }, { shape: [2, 2] });
        const c = gpu.add(a, b);
        await c.load();
        assertAlmostEqual(c, [2, 3, 4, 5]);
    }],
    ["a - b", async () => {
        const a = gpu.full(2, { shape: [2, 2] });
        const b = gpu.arange({ stop: 4 }, { shape: [2, 2], dtype: "f32" });
        const c = gpu.sub(a, b);
        await c.load();
        assertAlmostEqual(c, [2, 1, 0, -1]);
    }],
    ["a * b", async () => {
        const a = gpu.full(2, { shape: [2, 2] });
        const b = gpu.arange({ stop: 4 }, { shape: [2, 2], dtype: "f32" });
        const c = gpu.mul(a, b);
        await c.load();
        assertAlmostEqual(c, [0, 2, 4, 6]);
    }],
    ["a / b", async () => {
        const a = gpu.full(2, { shape: [2, 2] });
        const b = gpu.arange({ stop: 4 }, { shape: [2, 2], dtype: "f32" });
        const c = gpu.div(a, b);
        await c.load();
        assertAlmostEqual(c, [2/0, 2, 1, 2/3]);
    }],
    ["a (f32) + b (u32)", async () => {
        const a = gpu.full(2.3);
        const b = gpu.full(1, { dtype: "u32" });
        const c = gpu.add(a, b);
        await c.load();
        assertEqual(c.dtype, "f32");
        assertAlmostEqual(c, [3.3]);
    }],
]);

TEST("f(a)", [
    ["sin(a)", async () => {
        const a = gpu.arange({ start: 1, step: 0.2, stop: 2 }, { dtype: "f32" })
        const b = gpu.sin(a);
        await b.load();
        assertAlmostEqual(
            b,
            [Math.sin(1), Math.sin(1.2), Math.sin(1.4), Math.sin(1.6), Math.sin(1.8)],
            { rtol: 1e-4 },
        );
    }],
    ["floor(a)", async () => {
        const a = gpu.arange({ start: 1, step: 0.2, stop: 2 }, { dtype: "f32" });
        const b = gpu.floor(a);
        await b.load();
        assertAlmostEqual(b, [1, 1, 1, 1, 1]);
    }],
]);


TEST("f(a, b)", [
    ["max(a, b)", async () => {
        const a = gpu.arange({ start: 1, step: 0.5, stop: 3 }, { dtype: "f32" });
        const b = gpu.full(2, { shape: [4] });
        const c = gpu.min(a, b);
        await c.load();
        assertAlmostEqual(c, [1, 1.5, 2, 2]);
    }],
]);


TEST("log(a)",[
    ["log(0)", async () => {
        const a = gpu.Array();
        const b = gpu.log(a);
        await b.load();
        assertAlmostEqual(b, [Math.log(0)]);
    }],
    ["log(1)", async () => {
        const a = gpu.ones();
        const b = gpu.log(a);
        await b.load();
        assertAlmostEqual(b, [Math.log(1)]);
    }],
    ["log(2)", async () => {
        const a = gpu.full(2);
        const b = gpu.log(a);
        await b.load();
        assertAlmostEqual(b, [Math.log(2)]);
    }],
]);


TEST("pow(a, b)", [
    ["pow(0, 0)", async () => {
        const a = gpu.Array();
        const b = gpu.Array();
        const c = gpu.pow(a, b);
        await c.load();
        assertAlmostEqual(c, [0 ** 0]);
    }],
    ["pow(1, 0)", async () => {
        const a = gpu.full(1);
        const b = gpu.full(0);
        const c = gpu.pow(a, b);
        await c.load();
        assertAlmostEqual(c, [1 ** 0]);
    }],
    ["pow(0, 1)", async () => {
        const a = gpu.full(0);
        const b = gpu.full(1);
        const c = gpu.pow(a, b);
        await c.load();
        assertAlmostEqual(c, [0 ** 1]);
    }],
]);


