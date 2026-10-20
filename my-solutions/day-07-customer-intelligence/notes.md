







## Outputs: 
Segment Profile:
              BusinessSegment  Customers  AvgRecency  AvgFrequency  AvgMonetaryValue  ...  AvgDaysBetweenOrders  AvgCancelledInvoices  AvgCancellationRate  CustomerShare  RevenueShare
2            Loyal Customers       1276   13.476489      9.103448       4870.256144  ...             55.886731              1.873824             0.138538       0.291857      0.697978
0                  Champions        898   96.316258      4.340757       1907.777964  ...             72.494484              0.885301             0.133084       0.205398      0.192417
5        Potential Loyalists        974  160.206366      1.578029        420.552066  ...             95.992687              0.224846             0.076303       0.222781      0.046006
1  Inactive / Low Engagement        580   61.270690      1.131034        463.508741  ...            139.797021              0.186207             0.077299       0.132662      0.030194
4                      Other        549  201.879612      1.417476        479.297437  ...            114.085238              0.242259             0.133340       0.125572      0.029554
3              New Customers         95   23.042105      1.000000        360.910316  ...                   NaN              0.084211             0.042105       0.021729      0.003851

[6 rows x 15 columns]

Cluster Profiles:
    Cluster  Customers  AvgRecency  AvgFrequency  AvgMonetaryValue  ...  AvgDaysBetweenOrders  AvgCancelledInvoices  AvgCancellationRate  CustomerShare  RevenueShare
1        1       1750   37.071429      7.858857       3939.677126  ...             88.469797              1.552000             0.125610       0.400274      0.772648
2        2       1299   52.167180      1.916025        664.661897  ...             40.539218              0.056197             0.010701       0.297118      0.096759
4        4        492  115.193900      2.145969       1068.246473  ...             58.841559              1.603659             0.481982       0.112534      0.058901
0        0        829  260.113390      1.572979        475.345657  ...             50.741517              0.089264             0.025274       0.189616      0.044162
3        3          2  163.500000      1.500000     122828.050000  ...            204.974306              1.000000             0.416667       0.000457      0.027530

[5 rows x 15 columns]

BusinessSegment x Cluster (counts):
 Cluster                      0    1    2  3    4
BusinessSegment                                 
Champions                  122  539  116  0  121
Inactive / Low Engagement   19   28  449  0   84
Loyal Customers              0  980  233  1   62
New Customers                0    0   87  0    8
Other                      311   65   77  1   95
Potential Loyalists        377  138  337  0  122

BusinessSegment x Cluster (% within segment):
 Cluster                        0      1      2      3      4
BusinessSegment                                             
Champions                  0.136  0.600  0.129  0.000  0.135
Inactive / Low Engagement  0.033  0.048  0.774  0.000  0.145
Loyal Customers            0.000  0.768  0.183  0.001  0.049
New Customers              0.000  0.000  0.916  0.000  0.084
Other                      0.566  0.118  0.140  0.002  0.173
Potential Loyalists        0.387  0.142  0.346  0.000  0.125

High-value overlap summary:
 {'top_business_segment': 'Loyal Customers', 'top_cluster': 1, 'top_segment_size': 1276, 'top_cluster_size': 1750, 'overlap_customers': 980, 'overlap_pct_of_segment': 76.8, 'overlap_pct_of_cluster': 56.0}