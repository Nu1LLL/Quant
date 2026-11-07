# Pre-result clarification: deterministic event reservation

Parent preregistration: `2026-11-08_binance_new_perpetual_post_launch_short.md`, commit `d2b23e0`. This clarification is committed before any formal event return, PnL or Gate metric is read. It does not alter the signal direction, asset set, price, cost, size, stop, horizon or threshold.

To make baseline and both stress scenarios use **identical event selection** with one position at a time, an eligible event reserves the single trading slot from its original `t0+30m` candidate entry until its **scheduled `t0+24h` exit**, even if the protective stop closes it sooner. Every later candidate whose original entry is before that reserved expiry is skipped as an overlap. A capacity-failed event does not reserve a slot. This rule is decided entirely from earlier event timestamps and the current entry's completed six-bar volume; it cannot use later stop or price outcomes. The one-bar-latency stress enters five minutes later within the already reserved slot. Exact ties are broken lexically.

Daily end marks use the completed 23:55 UTC bar close; events may trade sequentially within one day, but never occupy overlapping reserved slots. No raw archive, event return, trade PnL or Gate result was consulted for this clarification.
