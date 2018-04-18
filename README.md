# tensowflow_ddpg
## Deep Deterministic Policy Gradient
DDPG method is RL method that utilizes DPG Theorem, that can handle continuous action space. This implementation depends on Tensowflow and this RL model is experimented on aigym.
Following papers are referred when implementing this model.

- https://arxiv.org/pdf/1509.02971.pdf
- https://arxiv.org/pdf/1511.04143.pdf

## How to use
```
git clone https://github.com/sykwer/tensorflow_ddpg.git
cd tensowflow_ddpg
python main.py
```

## Experiment
Experiment on InvertedPendulum-v2

![InvertedPendulum-v2_experiment](https://github.com/sykwer/tensorflow_ddpg/blob/master/images/InvertedPendulum-v2_experiment.png)
(episode - total reward)

## Resources
If you can conprehend Japanese article, refer to following articles! I have written several articles to understand DDPG theorem.

- https://sykwer.hatenablog.jp/entry/2018/03/08/105711
- https://sykwer.hatenablog.jp/entry/2018/03/10/Understand_DDPG_step_by_step_%282%29
