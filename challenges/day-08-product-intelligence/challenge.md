# Challenge 03 — Product & Basket Intelligence

## Scenario

The retailer now understands its customer base, but customer value alone does not explain what customers actually buy together.

The commercial team wants to improve:

- cross-selling
- recommendations
- merchandising
- product bundles

Your task is to analyze relationships between products and identify purchasing patterns that could support these decisions.

---

## Business Question

**Which products are purchased together, and how can basket behaviour support better cross-selling and merchandising decisions?**

---

## Tasks

### 1. Analyze product performance

Profile products using measures such as:

- units sold
- revenue
- number of orders
- number of customers
- average quantity per purchase

Distinguish product popularity from product value.

---

### 2. Build transaction baskets

Transform purchase data into invoice-level baskets.

Each basket should represent the products purchased within one successful order.

Exclude cancellation transactions from basket construction.

Consider whether extremely rare products should be removed before association analysis.

---

### 3. Analyze basket characteristics

Investigate:

- basket size
- number of unique products per order
- distribution of basket sizes
- proportion of single-product orders

Determine whether cross-selling analysis is relevant for a meaningful proportion of transactions.

---

### 4. Mine frequent itemsets

Use an association-rule approach such as:

- Apriori
- FP-Growth

Identify frequently co-purchased products.

Experiment with appropriate minimum-support thresholds.

---

### 5. Generate association rules

Evaluate product relationships using:

**Support**
- How frequently does the combination occur?

**Confidence**
- Given product A was purchased, how often was product B also purchased?

**Lift**
- Are A and B purchased together more often than expected if their purchases were independent?

Avoid treating high confidence alone as evidence of a strong relationship.

---

### 6. Identify commercially useful relationships

Look for:

- high-lift product pairs
- high-confidence cross-sell opportunities
- frequently purchased combinations
- relationships involving high-revenue products

Translate rules into actual product descriptions where possible.

---

### 7. Connect products to customer segments

Join customer segments from Challenge 02 back to transaction data.

Investigate whether different customer groups purchase different:

- products
- categories
- basket combinations

Determine whether cross-selling recommendations could be segment-specific.

---

### 8. Optional — Build a product network

Represent:

- products as nodes
- co-purchase relationships as edges

Use association strength to identify:

- product communities
- highly connected products
- potential anchor products

Keep the visualization interpretable rather than attempting to display every product.

---

## Deliverables

Produce:

1. Product-performance analysis.
2. Basket-level dataset.
3. Basket-size analysis.
4. Frequent itemsets.
5. Association rules.
6. A shortlist of commercially interesting product relationships.
7. Segment-specific product insights.
8. Optional product-network visualization.
9. Cross-selling or merchandising recommendations.

---

## Final Questions

Answer:

1. Which products drive volume and revenue?
2. Which products are frequently purchased together?
3. Which associations are genuinely stronger than chance?
4. Which rules appear commercially actionable?
5. Do customer segments exhibit different basket behaviour?
6. How could the retailer use these findings?