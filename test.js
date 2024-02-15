import van from "https://cdn.jsdelivr.net/gh/vanjs-org/van/public/van-latest.min.js"

const {
    div, pre,
    details, summary,
    table, thead, tbody, tr, th, td,
} = van.tags;

const TEST_CONFIG = {
    running: van.state("⏳"),
    success: van.state("✔"),
    failure: van.state("❌"),
    unknown: van.state("❓"),
};


const almostEqual = (x, y, tol) => {
    const atol = tol?.atol ?? 1e-6;
    const rtol = tol?.rtol ?? 1e-5;

    if(Object.is(x, y)){
        return true;
    }

    const diff = Math.abs(x - y);
    if(diff < atol){
        return true;
    }
    if(diff < Math.abs(x * rtol)){
        return true;
    }
    return false;
};

const assertAlmostEqual = (x, y, tol) => {
    if((typeof x === "object") &&
       (typeof y === "object") &&
       (x[Symbol.iterator] !== undefined) &&
       (y[Symbol.iterator] !== undefined)){
        const xit = x[Symbol.iterator]();
        const yit = y[Symbol.iterator]();

        while(true){
            let { value: xvalue, done: xdone } = xit.next();
            let { value: yvalue, done: ydone } = yit.next();

            xdone ??= false;
            ydone ??= false;

            if(xdone && ydone){ return; }
            if(xdone ^ ydone){
                throw new Error(`Fail Almost Equal: Wrong Length`);
            }

            assertAlmostEqual(xvalue, yvalue, tol);
        }
    }

    if(!almostEqual(x, y, tol)){
        throw new Error(`Fail Almost Equal: ${x} !~ ${y}`);
    }
};

const assertEqual = (x, y) => {
    if((typeof x === "object") &&
       (typeof y === "object") &&
       (x[Symbol.iterator] !== undefined) &&
       (y[Symbol.iterator] !== undefined)){
        const xit = x[Symbol.iterator]();
        const yit = y[Symbol.iterator]();

        while(true){
            let { value: xvalue, done: xdone } = xit.next();
            let { value: yvalue, done: ydone } = yit.next();

            xdone ??= false;
            ydone ??= false;

            if(xdone && ydone){ return; }
            if(xdone ^ ydone){
                throw new Error(`Fail Equal: Wrong Length`);
            }

            assertEqual(xvalue, yvalue);
        }
    }

    if((x !== y) && !(Number.isNaN(x) && Number.isNaN(y))){
        throw new Error(`Fail Equal: ${x} !== ${y}`);
    }
};

const Run = async (f, result, detail) => {
    try {
        await f();
        result.val = true;
    } catch(e){
        result.val = false;
        detail.val = `${e.name}: ${e.message}`;
    }
}

const TestCase = ([name, f], result) => {
    const detail = van.state("");

    Run(f, result, detail);
    return tr(
        td(name),
        td(() => {
            switch(result.val){
            case null:
                return TEST_CONFIG.running.val;
            case true:
                return TEST_CONFIG.success.val;
            case false:
                return TEST_CONFIG.failure.val;
            default:
                return TEST_CONFIG.unknown.val;
            }
        }),
        td(pre(detail)),
    );
};

const TEST = (summaryLine, cases) => {
    const results = cases.map(() => van.state(null));
    const sall = van.derive(() => {
        return results.every(r => r.val) ?
            TEST_CONFIG.success.val:
            TEST_CONFIG.failure.val;
    });
    const scount = van.derive(() => results.reduce((a, r) => r.val + a, 0));

    const test = details(
        summary(() => `${sall.val}: ${summaryLine}: ${scount.val} / ${cases.length}`),
        table(
            thead(tr(
                th({scope: "col"}, "name"),
                th({scope: "col"}, "result"),
                th({scope: "col"}, "detail"))),
            tbody(cases.map((c, i) => TestCase(c, results[i]))),
        ),
    );

    van.add(document.body, test);
};


export { TEST, assertEqual, assertAlmostEqual };
