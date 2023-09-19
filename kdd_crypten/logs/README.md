This directory contains logs from running KDDSortingNetwork.py, KDDSortingNetworkRevealed.py, and KDDMergeSort.py on AWS servers. The file name contains the information about what Crypten computation each log was obtained from.

| Prefix | Python Program |
| :--- | :--- |
| d | KDDSortingNetwork.py |
| b | KDDSortingNetworkRevealed.py |
| m | KDDMergeSort.py |

| First Number | Number of Users |
| :--- | :--- |
| 1000 | 1000 users |
| 10000 | 10000 users |

| Second Number | Number of Websites |
| :--- | :--- |
| 600 | 600 websites |
| (omitted) | 60 websites |

KDDMergeSort.py is the fastest. It runs about twice as fast as KDDSortingNetworkRevealed with 10000 users and 600 websites.
KDDSortingNetworkRevealed.py is the second fastest.
KDDSortingNetwork.py is the slowest.

The timing information, which is contained in the logs, is also displayed here. The table contains the number of seconds to complete one iteration of KDD Algorithm 1.

| Dataset size | KDDSortingNetwork.py | KDDSortingNetworkRevealed.py | KDDMergeSort.py |
| :--- | :--- | :--- | :--- |
| 1000 users, 60 websites | 92 | 3518 | 3630 |
| 10000 users, 60 websites | 65 | 2542 | 2484 |
| 10000 users, 600 websites | 63 | was not run | 1235 |
