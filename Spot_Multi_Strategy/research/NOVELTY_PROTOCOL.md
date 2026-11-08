# Economic-mechanism novelty check before every future experiment

Before writing a new preregistration, inspect **all tracked** legacy
`reports/*/PREREGISTRATION.md`, their `RESULT.md`/`REPORT.md` summaries, and newer
`research/preregistrations/*.md`. Use the read-only helper from the repository
root, for example:

```sh
PYTHONPATH=Spot_Multi_Strategy python3 -m research.novelty_inventory \
  --repo . --family real_yield \
  --exclude Spot_Multi_Strategy/research/preregistrations/NEW.md
```

For another mechanism, supply broad `--phrase` synonyms covering the economic
channel, source series names, provider names, asset/market terminology and
Chinese/English descriptions. Read the *full* prior registrations surfaced by
the search. A zero text-match count does **not** prove economic independence;
compare the new proposed causal variable and payoff against the inventory of
all titles. State the collision-screen terms, relevant prior experiments and
why the proposed information source differs in the preregistration **before
the separate preregistration commit**. If the same causal mechanism already
has a sealed test, classify a new instrument or provider as a related
implementation variant rather than a new Alpha family; do not run an
unregistered variation on the same history.

This procedure was added after the FRED DFII10 BTC/ETH experiment repeated
the prior official-Treasury TIPS 20-observation direction signal. The
[correction](results/2026-11-08_real_yield_duplicate_mechanism_erratum.md)
preserves both sealed results and describes the mistake.
