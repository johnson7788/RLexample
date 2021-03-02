# 强化学习的简单示例

## Installing Anaconda and OpenAI gym

* Download and install Anaconda [here](https://www.anaconda.com/download)
* Install OpenAI gym, 下面2个都要安装 
```
pip install gym
pip install gym[atari]
```

## Examples

* gym的环境操作示例
```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```

* 随机示例 ```CartPole-v0```

```
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
env.close()
```

* 随机示例 (```Pong-ram-v0```,```Acrobot-v1```,```Breakout-v0```)

```
python my_random_agent.py Pong-ram-v0
```

* 非常简单的智能体示例 ```CartPole-v0``` or ```Acrobot-v1```

```
python my_learning_agent.py CartPole-v0

```

* 使用CPU玩乒乓球游戏 (with a great [blog](http://karpathy.github.io/2016/05/31/rl/)). 
使用预训练模型 ```pong_model_bolei.p```(经过 20,000 回合的训练), 
使用pg-pong.py文件加载 [save_file](https://github.com/metalbubble/RLexample/blob/master/pg-pong.py#L15) in the script. 

```
python pg-pong.py

```

* Random navigation agent in [AI2THOR](https://github.com/allenai/ai2thor)

```
python navigation_agent.py
```

