# Data Quality Audit

- Shape: 541,909 rows x 11 columns
- Date range: 2010-12-01 08:26:00 to 2011-12-09 12:50:00

## Missing values
```
InvoiceNo              0
StockCode              0
Description         1454
Quantity               0
InvoiceDate            0
UnitPrice              0
CustomerID        135080
Country                0
Revenue                0
IsCancellation         0
NonProduct             0
```

- Duplicate rows: 5,268
- Missing CustomerID rows: 135,080
- Missing Description rows: 1,454
- Quantity <= 0 rows: 10,624
- UnitPrice <= 0 rows: 2,517
- Cancellation rows: 9,288
- Non-product code rows: 2,912

## Analysis impact
- Missing CustomerID affects only customer-level analysis.
- Cancellation rows are preserved for net revenue and cancellation analysis.
- Non-positive quantity/price and non-product codes are excluded from valid purchases.
- Duplicates are reported; not removed blindly.