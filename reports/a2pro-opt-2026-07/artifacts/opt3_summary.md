# OPT-3 Federated Optimization Demo

| method | initial loss | final loss | improvement | points | final time | mean staleness | max staleness |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg-5clients | 11.557475 | 0.001011 | 11430.54x | 40 |  |  |  |
| FedAvg-10clients | 11.557475 | 0.001032 | 11201.83x | 40 |  |  |  |
| FedAvg-20clients | 11.557475 | 0.001022 | 11309.82x | 40 |  |  |  |
| AsyncSGD | 11.557475 | 1.074178 | 10.76x | 160 | 71 | 6.83 | 16 |
| Async-LocalSGD | 11.557475 | 0.003963 | 2916.08x | 100 | 49 | 6.72 | 14 |
