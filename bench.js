import van from "https://cdn.jsdelivr.net/gh/vanjs-org/van/public/van-latest.min.js"
import { assertAlmostEqual } from "./test.js";

const {
    details, summary,
    table, thead, tbody, tr, td, th,
} = van.tags;


const BENCH = async (SummaryLine, cases) => {
    const b = tbody();

    const bench = details(
        summary(SummaryLine),
        table(
            thead(tr(th({scope: "col"}, "name"), th({scope: "col"}, "elapsed"))),
            b,
        ),
    );

    let ret = null;
    for(const c of cases){
        const [name, f] = c;

        const t1 = performance.now();
        const R = await f();
        const t2 = performance.now();
        ret ??= R;
        assertAlmostEqual(R, ret);

        van.add(b, tr(th({scope: "row"}, name),
                      td(`${(t2 - t1).toFixed(2)}ms`)));
    }
    van.add(document.body, bench);
};


export { BENCH };
