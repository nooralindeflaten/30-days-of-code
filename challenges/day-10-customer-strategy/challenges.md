# Challenge 05 — From Customer Prediction to Retention Strategy

## Scenario

The retention model can estimate which customers are likely to become inactive.

However, contacting every high-risk customer is neither realistic nor necessarily valuable.

A customer with a 90% inactivity probability but very little expected future value may be less important than a highly valuable customer with a 50% inactivity probability.

Management therefore wants the analysis translated into a prioritization strategy.

---

## Business Question

**Which customers should the retailer prioritize for retention, and where is the greatest customer value at risk?**

---

## Tasks

### 1. Combine risk and customer value

For each customer, combine:

- predicted inactivity probability
- historical or expected customer value

Create an initial prioritization measure such as:

`Value at Risk = P(Inactive) × Customer Value`

Clearly document the assumptions behind the value measure.

---

### 2. Compare targeting strategies

Assume the company can contact only a fixed proportion of customers.

Compare strategies such as:

- random targeting
- highest inactivity risk
- highest customer value
- highest value at risk

Determine which strategy concentrates the greatest relevant customer value.

---

### 3. Create customer priority groups

Develop an interpretable framework based on risk and value.

For example:

| Customer Value | Inactivity Risk | Strategy |
|---|---|---|
| High | High | Immediate retention priority |
| High | Low | Protect / loyalty |
| Low | High | Low-cost intervention |
| Low | Low | Minimal intervention |

Justify the final strategy using the data.

---

### 4. Optional — Estimate future customer value

Go beyond historical Monetary value by investigating probabilistic Customer Lifetime Value.

Possible methods:

- BG/NBD for expected future purchase frequency
- Gamma-Gamma for expected transaction value

Estimate future customer value over a defined horizon.

Document the assumptions required by the models.

---

### 5. Calculate expected value at risk

If CLV is modeled, combine:

`Expected Value at Risk = P(Inactive) × Expected CLV`

Rank customers according to potential economic exposure.

---

### 6. Simulate a retention campaign

Create a hypothetical campaign scenario.

Define assumptions such as:

- number of customers that can be contacted
- contact cost
- intervention success probability
- retained customer value

Compare targeting strategies under the same campaign budget.

Clearly label intervention effects and costs as scenario assumptions.

They are not causal estimates derived from the observational dataset.

---

### 7. Connect customer and product intelligence

Use insights from Challenge 03 to suggest how interventions might differ.

Examples:

- segment-specific recommendations
- complementary-product offers
- reactivation offers based on historical product preferences
- VIP treatment for high-value customers

Do not claim these interventions will cause retention without experimental evidence.

---

### 8. Design the next experiment

Propose a randomized retention experiment.

Define:

- treatment population
- control group
- intervention
- primary outcome
- evaluation period
- business success metric

Explain how this experiment would allow the business to move from prediction toward causal decision-making.

---

## Deliverables

Produce:

1. Customer risk ranking.
2. Customer-value analysis.
3. Risk × value prioritization.
4. Comparison of targeting strategies.
5. Customer priority matrix.
6. Optional CLV model.
7. Retention-campaign simulation.
8. Proposed randomized experiment.
9. Final stakeholder recommendations.

---

## Final Questions

Answer:

1. Are the highest-risk customers also the most valuable?
2. Which customers represent the greatest value at risk?
3. Does risk-based targeting outperform random targeting?
4. Does combining risk and value improve prioritization?
5. How should retention strategy differ between customer groups?
6. What assumptions does the campaign simulation depend on?
7. What experiment should the business run next?

---

## Final Principle

Prediction answers:

**"Who is likely to become inactive?"**

Customer value answers:

**"Who matters economically?"**

Decision science asks:

**"Given limited resources, who should we act on?"**

Causal experimentation is then required to answer:

**"Did our intervention actually change customer behaviour?"**