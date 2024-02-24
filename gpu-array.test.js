import { createGPU } from "./gpu-array.js";
import {
    TEST,
    assertEqual, assertAlmostEqual,
    assertTruthy, assertFalsy,
    assertThrow,
} from "./test.js";


const gpu = await createGPU();


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


TEST("Array Op", [
    ["set()", async () => {
        const a = gpu.Array();
        assertFalsy(a.cpu_dirty);
        a.set(1, 0);
        assertTruthy(a.cpu_dirty);
    }],
    ["send()", async () => {
        const a = gpu.full(2, { shape: [3, 4] });
        assertTruthy(a.cpu_dirty);
        a.send();
        assertFalsy(a.cpu_dirty);
    }],
    ["reshape()", async () => {
        const a = gpu.full(1, { shape: [2, 3] });
        assertEqual(a.shape, [2, 3]);
        a.reshape([3, 2]);
        assertEqual(a.shape, [3, 2]);
        assertThrow(() => a.reshape([5, 5]));
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
    ["a + b (Broadcast)", async () => {
        const a = gpu.full(1.7, { shape: [2, 3] });
        const b = gpu.full(0.3);
        const c = gpu.add(a, b);
        await c.load();
        assertEqual(c.length, 6);
        assertAlmostEqual(c, [2, 2, 2, 2, 2, 2]);
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


TEST("Reduction Op", [
    ["sum (small)", async () => {
        const a = gpu.full(2, { shape: [32] });
        const b = gpu.sum(a);
        await b.load();
        assertAlmostEqual(b, [64]);
    }],
    ["sum (large)", async () => {
        const a = gpu.full(2, { shape: [300] });
        const b = gpu.sum(a);
        await b.load();
        assertAlmostEqual(b, [600]);
    }],
    ["prod", async () => {
        const a = gpu.full(2, { shape: [4] });
        const b = gpu.prod(a);
        await b.load();
        assertAlmostEqual(b, [16]);
    }],
]);


TEST("Reduce Func", [
    ["maximum (small)", async () => {
        const a = gpu.arange({ stop: 5 }, { dtype: "f32" });
        const b = gpu.maximum(a);
        await b.load();
        assertAlmostEqual(b, [4]);
    }],
    ["maximum (large)", async () => {
        const a = gpu.arange({ stop: 200 }, { dtype: "f32" });
        const b = gpu.maximum(a);
        await b.load();
        assertAlmostEqual(b, [199]);
    }],
]);


TEST("f16", [
    ["basic", async () => {
        const a = gpu.Array({ shape: [2], dtype: "f16" });
        a.set(1.2, 0);
        assertAlmostEqual(a, [1.2, 0], { rtol: 1e-3 });
    }],
    ["add", async () => {
        const a = gpu.ones({ shape: [2], dtype: "f16" });
        const b = gpu.arange({ stop: 2 }, { dtype: "f16" });
        const c = gpu.add(a, b);
        await c.load();
        assertAlmostEqual(c, [1, 2]);
    }],
    ["Type Promotion", async () => {
        const a = gpu.full(2.0, { shape: [2] });
        const b = gpu.full(2.0, { shape: [2], dtype: "f16" });
        const c = gpu.mul(a, b);
        await c.load();
        assertAlmostEqual(c, [4, 4]);
    }],
]);


TEST("Xoshiro128++", [
    ["u32", async () => {
        const prng = gpu.Xoshiro128pp({ seed: 0, size: 1 });
        const u32 = prng.next();
        assertEqual(u32.dtype, "u32");
        assertEqual(u32.shape, [1]);

        const a = await u32.get(0);
        const b = await prng.next().get(0);
        assertTruthy(a !== b);
    }],
    ["no seed", async () => {
        const prng = gpu.Xoshiro128pp({ size: 1 });
        const u32 = prng.next();
        assertEqual(u32.dtype, "u32");
        assertEqual(u32.shape, [1]);
    }],
    ["same seed", async () => {
        const size = 400;
        const p1 = gpu.Xoshiro128pp({ size, seed: 20 });
        const p2 = gpu.Xoshiro128pp({ size, seed: 20 });

        const u1 = p1.next();
        const u2 = p2.next();
        await Promise.all([u1.load(), u2.load()]);
        assertAlmostEqual(u1, u2);

        const f1 = p1.next("f32");
        const f2 = p2.next("f32");
        await Promise.all([f1.load(), f2.load()]);
        assertAlmostEqual(f1, f2);

        const s = gpu.sum(f1);
        assertAlmostEqual((await s.get(0)) / f1.length, 0.5, { rtol: 0.1 });
    }],
    ["norm", async () => {
        const size = 1000;
        const prng = gpu.Xoshiro128pp({ size });
        const X = prng.normal();
        await X.load();
        assertEqual(X.shape, [size]);

        const mu = gpu.div(gpu.sum(X), gpu.full(X.shape[0]));
        assertAlmostEqual(await mu.get(0), 0.0, { rtol: 0.1, atol: 0.1 });
    }],
]);


TEST("where", [
    ["simple", async () => {
        const c = gpu.arange({ start: 0, stop: 2 }, { dtype: "u32" });
        const T = gpu.full(1, { shape: [2] });
        const F = gpu.full(2, { shape: [2] });

        const w = gpu.where(c, T, F);
        await w.load();
        assertAlmostEqual(w, [2, 1]);
    }],
    ["broadcast", async () => {
        const c = gpu.arange({ start: 0, stop: 2 }, { dtype: "u32" });
        const T = gpu.full(1, { shape: [1] });
        const F = gpu.full(2, { shape: [1] });

        const w = gpu.where(c, T, F);
        await w.load();
        assertAlmostEqual(w, [2, 1]);
    }],
]);
