# RedTeam Hardening Contract

The durable RedTeam report lives in `src/redteam_report.py` and is generated
into:

- `output/data/redteam_report.json`
- `output/data/redteam_report.md`
- `output/figures/redteam_risk_register.png`

The findings are not broad criticism. Each row names:

- the vector under attack;
- a false-certification path;
- a negative-control input;
- a minimal remediation;
- the validation check that now guards the issue;
- the remediation status.

The validator fails if the report is missing, if an expected finding is absent,
if findings are not marked remediated, or if validation-check names are missing.
This keeps the critique part of the reproducible artifact surface rather than a
conversation-only note.

The current register also covers trace semantic drift, RxInfer companion
artifact drift, scholarship stage laundering, roadmap package overclaiming, and
claim-register closure, as well as the absence of a diagnostic comparison
figure. It also covers stale or replaced generated artifacts through the
reproducibility manifest check. Those are false-certification paths where a
pipeline could still exit successfully while the scientific interpretation
became weaker.
