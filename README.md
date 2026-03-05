[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/7GDpIzou)
# AI Stats Lab

Topics:

- Random variables
- CDF probabilities
- Valid PDFs
- Exponential distribution
- Gaussian distribution
- Monte Carlo verification
- PDF plotting

Allowed libraries:

- numpy
- scipy
- matplotlib
- math

Do not change function names.

---

# Setup

```bash
pip install numpy scipy matplotlib pytest
```

---

# Run Autograder

```bash
pytest -q test_AIstats_lab.py
```

---

# Question 1 — CDF Probabilities

Given the CDF

F_X(x) = (1 - e^{-x})u(x)

Compute analytically:

1. P(X > 5)
2. P(X < 5)
3. P(3 < X < 7)

Then verify P(X>5) using Monte Carlo simulation.

Return both analytical and simulated values.

---

# Question 2 — PDF Validation and Plot

Consider the candidate PDF

f(x) = 2x e^{-x²},  x ≥ 0

Tasks:

1. Verify non-negativity
2. Compute integral from 0 to ∞
3. Determine if the function is a valid PDF
4. Plot the PDF on [0,3]

Return:

- integral value
- validity (True/False)

---

# Question 3 — Exponential Distribution

Let

X ~ Exp(λ = 1)

Compute:

1. P(X > 5)
2. P(1 < X < 3)

Then estimate both probabilities using Monte Carlo simulation.

Return analytical and simulated results.

---

# Question 4 — Gaussian Distribution

Let

X ~ N(10, 2²)

Tasks:

1. Standardize the variable
2. Compute P(X ≤ 12)
3. Compute P(8 < X < 12)

Then simulate 100000 samples and estimate the probabilities.

Return analytical and simulated results.
