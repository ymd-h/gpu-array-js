import van from "https://cdn.jsdelivr.net/gh/vanjs-org/van/public/van-latest.min.js"

const {
    div,
    table, thead, tbody, tr, th, td,
} = van.tags;

const TEST_CONFIG = {
    running: van.state("⏳"),
    success: van.state("✔"),
    failure: van.state("❌"),
    unknown: van.state("❓"),
};


const almostEqual = (x, y, atol, rtol) => {
    atol ??= 1e-6;
    rtol ??= 1e-5;

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

const assertAlmostEqual = (x, y, atol, rtol) => {
    if(!almostEqual(x, y, atol, rtol)){
        throw new Error(`Fail Almost Equal: ${x} !~ ${y}`);
    }
};

const assertEqual = (x, y) => {
    if(x !== y){
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

const TestCase = ([name, f]) => {
    const result = van.state(null);
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
        td(detail),
    );
};

const TEST = cases => {
    const summary = table(
        thead(tr(
            th({scope: "col"}, "name"),
            th({scope: "col"}, "result"),
            th({scope: "col"}, "detail"))),
        tbody(cases.map(TestCase)),
    );

    van.add(document.body, summary);
};


export { TEST, assertEqual, assertAlmostEqual };
