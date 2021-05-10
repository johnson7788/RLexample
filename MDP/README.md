## Solving FrozenLake using Policy Iteration and Value Iteration

## Introduction
[FrozenLake env](https://gym.openai.com/envs/FrozenLake-v0/) 在OpenAI中，Markov决策过程就是一个非常经典的例子。 请详细研究任务的动态。 然后，我们将执行值迭代和策略迭代以搜索最佳策略。 

* 加载FrozenLake环境并研究环境动态 :
```
    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    print(env.env.P)
    # it will show 16 states, in which state, there are 4 actions, for each action, there are three possibile states to go with prob=0.333
```

* Run value iteration on FrozenLake, 在FrozenLake上运行值迭代 
```
python frozenlake_vale_iteration.py
设定最大迭代次数: 100000, 停止条件: 1e-20
价值迭代在迭代中收敛, 在迭代1373次后收敛
这个策略的平均分数是:  0.74
```

* Run policy iteration on FrozenLake
```
python frozenlake_policy_iteration.py
策略迭代收敛, 在迭代4次后收敛.
这个策略的平均分数是:  0.68
```

* Switch to FrozenLake8x8-v0 for more challenging task.
