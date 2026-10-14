# Statistical interpretation

This is the maintained interpretation contract for protocol schema 2.2.0,
`active-assembly-open-roadmap-v2-2026-10-13`. The source of truth is
`src/scientific_protocol.py`; current summaries and inferential ledgers are in
`output/data/research_protocol.json` and `output/data/quotient_classifier.json`.
See [evaluation](evaluation.md) for producers and [scholarship](scholarship.md)
for reference roles. Historical review counts are not current-run results.

## Units and splits

| Surface | Unit and actual boundary | Interpretation |
| --- | --- | --- |
| Temporal corpus | Ten authored fixtures; a 4/2/4 design inventory, while supervised, typed and observation-only estimators use complete-fixture leave-one-out folds. | A fold excludes its scored fixture; its training complement may cross inventory partitions. Fixture identity is not independently sampled source lineage. |
| Quotient classifier | 64 generated transformation families, 64 rows per family; family identifiers do not cross the train/development/test split, with 16 test families. | Authored template lineages are shared and dependent. Identifier separation does not establish lineage-disjoint population generalization. |
| Fixed-model adequacy | Same-length sequences simulated from fixed `A/B/D`, with explicit seed and repetitions. | Central simulation intervals describe model adequacy; no parameter posterior or population sampling is involved. |
| External contracts | Pinned project/revision/source-contract rows and the declared project-held-out endpoint. | Static provenance and label coverage do not measure classifier efficacy or satisfy independently replayed acquisition/build. |
| Sensing | Declared finite channels, horizons, budgets and deterministic seeds; local timing has its own environment receipt. | Exact enumeration is scoped to that decision problem; timing is machine-specific. |

## Metrics and uncertainty

Balanced accuracy and macro-F1 are the quotient co-primary comparison metrics.
Brier score, fixed-bin ECE, reliability diagrams, selective risk and confusion
counts are additional diagnostics; a model without probability predictions has
no invented calibration score. Temporal score Wilson intervals are descriptive
because steps within authored fixtures are dependent [@brown2001binomial].

The quotient percentile bootstrap resamples whole held-out families, preserving
all rows in each selected family [@cameron2008clusterbootstrap]. Paired intervals use
the same family draws for both methods. The interval need not contain the
original point estimate; plots show the point and endpoints independently.
These finite-inventory intervals do not establish population coverage.
Fractions of bootstrap differences at or below zero are descriptive and must
never be labeled p-values or supplied to multiplicity correction.

## Conditional paired randomization

For each model/baseline pair, the randomization diagnostic independently swaps
the two methods' complete prediction vectors within each held-out family and
recomputes the pooled metric difference. Its null assumes exchangeability of
the two predictions at that grouping level. It is conditional on the frozen
inventory and does not establish that shared authored lineages are independent.

The one-sided tail includes ties with tolerance `1e-12`. With 1,000 local-seeded
Monte Carlo draws, the reported value is `(exceedances + 1) / (draws + 1)`
[@phipson2010permutation]. The receipt records the null, seed, draw count,
exceedances and raw values. Holm step-down adjustment covers all 42 declared
comparisons: three models × seven baselines × two metrics [@holm1979multiple].
Validation recomputes raw-value arithmetic and the full adjustment ledger.

Protocol 2.2 is a retrospective methods correction. Even an adjusted value
below the threshold cannot promote superiority: the recorded superiority
flags remain withheld. A new inferential comparison must freeze its endpoint,
split, null, stopping rule and practical margin prospectively and then obtain
independent evaluation. Do not describe this correction as preregistration.

## Controls and leakage scope

Training vocabularies and fitted supports use training rows only; held-out
values map to an explicit unknown token. Missing fields and unseen values are
distinct. Observation-only EM receives training observations, with phase labels
reserved for scoring; its ordered scaffold and reference initialization still
constrain interpretation.

Temporal shuffled-label scores compare fixed predictions with seeded full
truth permutations. The quotient label-scramble control is different: it refits
on scrambled held-out rows and evaluates those same rows, retaining each
family's class totals. That in-sample control can diagnose a finite artifact;
it cannot certify absence of family leakage or estimate out-of-sample null
performance. Family shuffles, metadata probes, near-duplicate checks and
compiler/architecture probes each expose only their declared attack surface.
The complete ledger and weak/unstable starts must stay visible.

No interval, p-value, accuracy, exact certificate or passing implementation gate
turns authored phase labels into discovered semantics, proves complete ISA
correspondence, or supplies independent replication.
