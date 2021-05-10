# -*- coding: utf-8 -*-
"""
使用Value-Iteration解决FrozenLake环境。
Updated 17 Aug 2020
"""
import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

def run_episode(env, policy, gamma = 1.0, render = False):
    """ 评估策略的好坏, 通过运行回合并返回总奖励。
    args:
    env: gym environment.
    policy: the policy to be used. [0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]
    gamma: discount factor.
    render: 是否显示图画

    returns:
    total reward: agent根据策略返回奖励的总和
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    #直到回合结束才退出，即一盘游戏结束
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100, render=True):
    """ 通过运行n次来评估策略。 求平均值
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = render)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """
    通过价值获取策略，动态规划求解
    :param v:  序列，长度16，
    :param gamma:
    :return: 最佳行动， 获取每个state下应该采取的行动 [0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]
    """
    #初始化侧率，根据所有状态
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        #初始化动作
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                #获取概率，下一个状态，奖励和done，done表示是否回合结束
                p, s_, r, _ = next_sr
                #动态规划
                q_sa[a] += (p * (r + gamma * v[s_]))
        #找出最佳策略
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """
    Value-iteration algorithm, 用动态规划的方法求解
    :param env:
    :param gamma:
    :return:
    """
    #初始化值函数, env.env.nS 是状态个数，这里是16个
    v = np.zeros(env.env.nS)
    max_iterations = 100000
    # 迭代停止条件
    eps = 1e-20
    print(f"设定最大迭代次数: {max_iterations}, 停止条件: {eps}")
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            # env.env.nA是action的个数，env.env.P[s][a]是这个状态下的这个action，返回的p, s_, r, _ 是概率，状态，奖励
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('价值迭代收敛, 在迭代%d次后收敛' %(i+1))
            break
    return v


if __name__ == '__main__':
    env_name = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    render = False
    env = gym.make(env_name)
    #折扣因子
    gamma = 1.0
    #最佳价值求解
    optimal_v = value_iteration(env, gamma);
    #最佳行动策略
    policy = extract_policy(optimal_v, gamma)
    #评估这个策略的分数
    policy_score = evaluate_policy(env, policy, gamma, n=1000, render=render)
    print('这个策略的平均分数是: ', policy_score)
