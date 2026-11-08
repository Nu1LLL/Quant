# WhaleWatcher news and claim stream isolation

Quant's previous combined event-feature vector required a fresh news source
and fresh BTC/ETH directional-claim sources at the same time. The live
WhaleWatcher journal currently has fresh event polls but unavailable claim
polls because the external claim feed is not configured. Treating the entire
combined vector as unavailable was correct for the combined signal, but also
discarded usable, time-stamped news observations.

The new news-only vector has 17 fixed category/source count fields. It uses
only actual event-source poll status and event changes captured in Quant's
append-only journal. It excludes claim records, prices, article text, URLs,
AI advice, returns and orders. A missing/stale event poll remains unavailable
instead of appearing as zero news; a delayed first import cannot appear as
a fresh news burst, and future-dated publication waits until its timestamp.
The original combined vector still requires all three source streams.
Readiness now reports news continuity separately while its joint five-year
source Gate remains unchanged and false.

Read-only local verification on 2026-11-08 found news stream fresh, combined
news/claims stream unavailable, news vector available with 17 schema fields,
and independent trading OOS false. No live content or private local journal
was copied into this public repository. Focused WhaleWatcher tests: 31
passed. The broader crypto black-box suite: 102 passed. This is
infrastructure for future preregistered hypotheses, not a measured Alpha,
five-year OOS result, executable signal or trading authorization.
