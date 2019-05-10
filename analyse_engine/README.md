# Scripts for movie analysis
## Scripts usage:
*1) film_clust_lda_prod.py*
```
python film_clust_lda_prod.py -view true -mgn 83 -cluster 1
```
```
Options description:
    '-view', '--view_cluster', type=strBool, help="Display chosen cluster", required=True	
    '-mgn', '--max_group_number', type=group_type, help="Display max number of cluster groups (< 84)", required=True
    '-cluster', type=cluster_type, help="Max cluster id num to test (< 5)", required=True
```
2) forecasting_words_prod.py
