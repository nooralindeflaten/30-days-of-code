# Challenge 01 — E-Commerce Business Diagnosis

## Scenario

You are working as a Data Scientist for a UK-based online retailer.

The company has accumulated approximately one year of transaction data containing purchases, products, customers, prices, cancellations and international sales.

Before developing customer models or recommending interventions, management wants a clear picture of how the business currently performs.

Your task is to transform raw transaction data into a reliable commercial overview and identify the most important patterns in revenue, customers, products and purchasing behaviour.

---

## Business Question

**What is driving the retailer's commercial performance, and where are the most important opportunities or risks?**

---

## Dataset

Use the **UCI Online Retail** dataset.

The dataset contains transaction-level records including:

- Invoice number
- Product / stock code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

Cancellation invoices can be identified by invoice numbers beginning with `C`.

---

## Tasks

### 1. Audit the raw data

Investigate:

- dataset dimensions
- data types
- missing values
- duplicate records
- missing customer identifiers
- missing product descriptions
- zero or negative quantities
- zero or negative prices
- cancellation transactions

Determine which issues affect different types of analysis.

Do not remove unusual observations without first understanding what they represent.

---

### 2. Define valid purchases

Create a clearly documented definition of a successful purchase transaction.

Separate:

- completed purchases
- cancellations / returns
- unusable or ambiguous records

Preserve cancellation information for later customer analysis.

---

### 3. Define business KPIs

Calculate relevant commercial metrics such as:

- gross revenue
- net revenue
- number of orders
- number of customers
- number of products sold
- units sold
- average order value
- items per order
- cancellation rate

Clearly distinguish gross purchase activity from net economic value after cancellations or returns.

---

### 4. Analyze revenue over time

Investigate revenue at appropriate time intervals.

Look for:

- growth or decline
- seasonality
- unusually strong or weak periods
- changes in order volume
- changes in average order value

Determine whether revenue changes are primarily driven by:

- more customers
- more orders
- larger orders
- higher-value products

---

### 5. Analyze geographic performance

Compare countries using measures such as:

- revenue
- customers
- orders
- average order value

Be cautious when interpreting countries with very small numbers of customers or transactions.

Determine how dependent the business is on its primary market.

---

### 6. Analyze product performance

Identify:

- highest-revenue products
- most frequently purchased products
- highest-volume products
- products with high cancellation activity

Consider whether the products generating the most units are also those generating the most revenue.

---

### 7. Investigate customer concentration

Calculate customer-level revenue contribution.

Determine whether revenue is evenly distributed across customers or concentrated among a relatively small group.

Create a cumulative revenue-contribution analysis.

---

## Deliverables

Produce:

1. A documented data-quality audit.
2. A clearly defined cleaning strategy.
3. A KPI summary.
4. Revenue-over-time analysis.
5. Geographic analysis.
6. Product-performance analysis.
7. Customer-concentration analysis.
8. A short executive summary of the most important commercial findings.

---

## Final Questions

Answer:

1. What currently drives revenue?
2. Is business performance stable over time?
3. How concentrated is revenue geographically?
4. Which products contribute most to performance?
5. How concentrated is revenue among customers?
6. How significant are cancellations?
7. Which findings deserve deeper investigation?

---

## Constraint

Do not build customer segments or predictive models yet.

The goal is to understand the business before introducing modeling.