# NSGA-2 for randomly wired neural networks
유전 알고리즘을 활용한 Randomly Wired Neural Network의 다중 목적 구조최적화


## Work with
- [김낙일](https://github.com/forestnoobie)


## Run
[1] Search
<pre>
<code>
 python main_search.py --params Main_experiment_1_search.json
</code>
</pre>
[2] Full training (=Architecture evaluation)
<pre>
<code>
 python main_full_training.py --params Main_experiment_1_full_training.json
</code>
</pre>


## Requirement
- python 3.5.2
- pytorch 1.1.0
- https://github.com/ildoonet/pytorch-gradual-warmup-lr
- https://github.com/ildoonet/cutmix


## Reference code
- https://github.com/seungwonpark/RandWireNN
- https://github.com/JiaminRen/RandWireNN
