# Challenge 02 — Customer Segmentation

## Scenario

The initial business analysis suggests that customers differ substantially in their purchasing behaviour and economic value.

The marketing team currently treats customers largely as one population.

Your task is to determine whether meaningful customer groups can be identified and whether those groups require different commercial strategies.

---

## Business Question

**Who are our most important customers, how do customer behaviours differ, and can those differences support meaningful segmentation?**

---

## Tasks

### 1. Build a customer-level dataset

Transform transaction-level data into one row per customer.

Engineer features describing customer behaviour.

At minimum consider:

- Recency
- Frequency
- Monetary value
- Average order value
- Items per order
- Product diversity
- Customer tenure
- Cancellation behaviour
- Purchase frequency / intervals

Clearly document how each feature is calculated.

---

### 2. Build RFM features

Calculate:

**Recency**
- Days since the customer's most recent successful purchase.

**Frequency**
- Number of successful purchase invoices.

**Monetary**
- Customer economic value after accounting for returns/cancellations.

Do not allow a cancellation to reset purchase recency or increase purchase frequency.

---

### 3. Create business-rule RFM segments

Score customers according to their:

- Recency
- Frequency
- Monetary value

Develop interpretable customer segments such as:

- Champions
- Loyal Customers
- Potential Loyalists
- New Customers
- At Risk
- Lost / Inactive

The exact segmentation logic should be justified rather than copied blindly from an existing framework.

---

### 4. Profile the business segments

For each segment investigate:

- customer count
- revenue contribution
- average monetary value
- average order value
- order frequency
- product diversity
- cancellation rate

Determine which groups contribute disproportionately to business value.

---

### 5. Build data-driven customer segments

Use clustering to investigate whether natural behavioural groups exist.

Consider K-Means.

Before clustering:

- inspect feature distributions
- address severe skew
- scale numerical features appropriately

Evaluate multiple values of `k` using methods such as:

- elbow analysis
- silhouette score
- cluster interpretability

---

### 6. Profile the clusters

Translate numerical clusters into understandable customer profiles.

For example:

`Cluster 0` is not a business insight.

Instead determine whether it represents something like:

> High-value, frequent, recently active customers with large baskets.

Compare cluster behaviour using:

- Recency
- Frequency
- Monetary
- Average order value
- product diversity
- cancellation rate
- tenure

---

### 7. Compare rule-based and data-driven segmentation

Compare RFM segments with clustering results.

Ask:

- Do both methods identify similar high-value customers?
- Does clustering reveal behaviours RFM misses?
- Which approach is easier for stakeholders to use?
- Which approach provides richer analytical information?

---

## Deliverables

Produce:

1. A customer-level feature dataset.
2. RFM analysis.
3. Business-rule customer segmentation.
4. Segment profiles.
5. A clustering model.
6. Cluster evaluation.
7. Cluster profiles.
8. Comparison of business-rule and ML segmentation.
9. Commercial recommendations for important customer groups.

---

## Final Questions

Answer:

1. Which customers generate the most value?
2. How concentrated is customer value?
3. What behavioural customer groups exist?
4. Do RFM and clustering tell the same story?
5. Which segmentation method is more actionable?
6. Which customer groups deserve different treatment?