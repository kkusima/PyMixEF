"""Minimal Gaussian random-intercept example."""

import pymixef

data = {
    "y": [1.0, 1.7, 2.4, 2.1, 2.9, 3.8],
    "time": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    "subject": ["A", "A", "A", "B", "B", "B"],
}

model = pymixef.Model.from_formula("y ~ time + (1 | subject)")
plan = model.compile(data, engine="lmm", method="reml")
print(plan.explain())
print(plan.fit().summary())
