# Scripts for movie analysis
## Scripts usage:
**1) film_clust_lda_prod.py**
```
python film_clust_lda_prod.py -cluster 1 -words hello world again fine
```
```
Options description:
    '-cluster', type=cluster_type, help="Max cluster id num to test (< 5)", required=True (will be removed)
	'-words', help="Words for films search (<= 4)",  default=[], nargs=4, required=True
```