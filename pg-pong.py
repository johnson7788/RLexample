# From http://karpathy.github.io/2016/05/31/rl/
""" 在Pong上训练具有（随机）策略梯度的agents。 使用OpenAI Gym。 """
import numpy as np
import pickle
import gym


def initial_model(save_file, resume=True):
    """
    初始化模型
    :param save_file: 模型的保存的文件
    :param resume:  bool是否恢复模型
    :return:
    """
    if resume:
        model = pickle.load(open(save_file, 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)
    return model


def sigmoid(x):
    """
    sigmoid 函数
    :param x:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def observation_extract(oberservation):
    """
    接收图像，并处理图像，下采样并过滤出乒乓球和球拍的位置
    接收形状为, [Height, weight, RGB],  210x160x3的输出的图像画面， uint8 frame into 6400 (80x80) 1D float vector
    :param I:
    :return:
    """
    I = oberservation
    # 裁剪高度，195-35=160, 取高度中间的160, 那样记分框就不会被截取到了，得到的形状是 [160,160,3]
    I = I[35:195]
    # 下采样，高和宽，步长为2采样一个像素，只取R通道，形状变为 [80,80]
    I = I[::2, ::2, 0]
    # 消除背景色，值等于144和109的都置为0，那么乒乓球和球拍的位置就显示出来了
    I[I == 144] = 0
    I[I == 109] = 0
    # 把乒乓球和球拍的设为1，
    I[I != 0] = 1
    # 压测成1维, 80x 80-->6400, 变成浮点数
    f = I.astype(np.float).ravel()
    return f


def discount_rewards(r):
    """
    采取一维浮动奖励数组并计算折扣奖励
    :param r: 一个回合的所有奖励， shape， (5746, 1)这里的5746是不确定的，因为不知道进行了多少次接收图像，然后后进行一次动作，
    这里是接收了5746次状态，进行了5746动作，然后游戏结束，或者我方式21分，或者对方是21分
    :return:
    """
    # 初始化一个折扣,全为0的，形状和输入r一样的形状
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # 重置总和，因为这是游戏边界（特定于乒乓球游戏！）
            running_add = 0  #
        # 计算出奖励
        running_add = running_add * gamma + r[t]
        # 第t个位置的奖励折扣
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    """
    根据收到的变化的x的状态，计算概率和隐藏神经元
    :param x: (6400,)
    :return:
    """
    h = np.dot(model['W1'], x)
    # Relu激活函数的非线性
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    # 采取行动2的返回概率，以及隐藏状态
    return p, h


def policy_backward(eph, epdlogp, epx):
    """
    向后传递。
    :param eph: eph是中间隐藏状态的数组
    :param epdlogp: logits归一化后的结果
    :param epx: 所有的输入状态，
    :return:
    """
    # eph shape (7559, 200) --> eph.T -->(200, 7559) ;  epdlogp (7559, 1),   ; dot后的结果 (200,1) , ravel压缩成1维， dW2 (200,)
    dW2 = np.dot(eph.T, epdlogp).ravel()
    # model['W2'], 200个数的数组,  外积, epdlogp shape (5448, 1),---> dh shape (5448, 200)
    dh = np.outer(epdlogp, model['W2'])
    # 反向传播， relu，小于0的置为0
    dh[eph <= 0] = 0
    #  dh.T shape (200,5448), epx shape (5448, 6400) -->  dW1 (200, 6400)
    dW1 = np.dot(dh.T, epx)
    # grad 也是设计成和model的格式相同，方便更新模型的参数
    grad = {'W1': dW1, 'W2': dW2}
    return grad


def initial_env(game="Pong-v0", verbose=False):
    """
    Pong-v0介绍： https://gym.openai.com/envs/Pong-v0/
    :return:
    """
    env = gym.make(game)
    if verbose:
        print(f"游戏环境初始化完成: {game}")
        # 即整个动画的画面，这里是 Box(0, 255, (210, 160, 3), uint8)，
        print('观测空间 = {}'.format(env.observation_space))
        # 可以采取的动作是6个, Discrete(6)
        print('动作空间 = {}'.format(env.action_space))
        # 0到255个像素的图画，low是全为0，(210, 160, 3), high全为255，(210, 160, 3),
        print('观测范围 = {} ~ {}'.format(env.observation_space.low,
                                      env.observation_space.high))
        # 动作数是6
        print('动作数 = {}'.format(env.action_space.n))
        #这里有意义的动作只有Right和Left，即对应2和3
        print(f'动作的意义是: {env.get_action_meanings()}')
    return env

def do_one_game(observation, verbose=False):
    """
    进行一局游戏
    :return:
    """
    # 一局结束
    done = False
    # 记录下当前乒乓球的分数，我方的分数, 和对方的分数
    myscore = 0
    comscore = 0
    #存储上一个frame的状态x
    prev_x = None
    # 这一小的回合接收了多少次参数
    ober_num = 0
    #这一局的reward合计
    reward_sum = 0
    #记录所有的输入状态, 所有的隐藏状态, logits, 所有奖励
    xs, hs, dlogps, drs = [], [], [], []
    while not done:
        if render:
            # 显示画面
            env.render()
        ober_num += 1
        # 对观测值进行预处理，将网络输入设置为不同图像, 即环境状态state， observation是原始画面
        cur_x = observation_extract(observation)
        # 经过简单处理后的的环境状态cur_x，x是变化的画面的状态
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        # 记录一下上一个状态
        prev_x = cur_x
        # 前向策略网络并根据返回的概率对操作进行采样
        aprob, h = policy_forward(x)
        # 根据计算得出的概率，判断我们要采取的行动
        # 如果不是测试，是训练，那么我们加入一个随机数，即探索和利用中的探索的概念, 2代表右移，3代表左移
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
        # 记录各种中间状态，需要反向传播
        # 观察状态 observation
        xs.append(x)
        # 隐藏状态
        hs.append(h)
        # 定义一个label
        y = 1 if action == 2 else 0  # a "fake label"
        # 记录下采取行的的概率
        dlogps.append(
            y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        # 把下一步要采取的行动提交给环境，例如这里action是3， 然后得到奖励和环境的观察结果
        observation, reward, done, info = env.step(action)
        # 累积奖励
        reward_sum += reward
        # 累积奖励记录下, 记录奖励
        drs.append(reward)
        # reward 不等于0，这里等于1或-1，表示得了1分或者输掉1分，游戏结束了， 一个小的回合
        if reward != 0:
            if reward == -1:
                # 对方+1分
                comscore += 1
            else:
                myscore += 1
            if verbose:
                print(f"第{myscore + comscore}次小回合，进行了{ober_num}次输入图像,分出了小的回合胜负, 奖励是: {reward}, 当前我方分数是{myscore},对方分数是{comscore}")
            # 重置下这个小的回合的接收图像次数
            ober_num = 0
    # 将本局的所有输入，隐藏状态，动作梯度和奖励堆叠在一起， 所有的形状是不同的
    # 所有的输入状态， shape, (5896, 6400)，  (8641, 6400)
    epx = np.vstack(xs)
    # 所有的隐藏状态, shape, (5896, 200)
    eph = np.vstack(hs)
    # 所有的logits, shape (5896, 1)
    epdlogp = np.vstack(dlogps)
    # 所有的奖励, (5896, 1),   (8641, 1)
    epr = np.vstack(drs)
    return epx, eph, epdlogp, epr, reward_sum, myscore, comscore

def do_train(verbose=False):
    grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # 更新在一个批次中添加梯度的缓冲区
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory
    observation = env.reset()
    running_reward = None
    for e in range(epoch):
        print(f"第{e}个epoch")
        #我方的胜利次数统计
        win = 0
        for b in range(batch_size):
            # 整个游戏，即一次游戏结束，谁先达到21分，谁胜利
            b += 1
            print(f"第{b}局开始:")
            epx, eph, epdlogp, epr, reward_sum, myscore, comscore = do_one_game(observation, verbose=verbose)
            print(f"第{b}局结束:")
            print(f"第{b}局的分数是: 我方总的奖励分数是{int(21 + np.sum(epr))}分, 当前我方分数是{myscore},对方分数是{comscore}")
            if myscore > comscore:
                win +=1
            # 计算折价奖励
            discounted_epr = discount_rewards(epr)
            # 归一化， 将奖励标准化为unit normal（帮助控制梯度估计量方差)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            # 利用梯度, logits归一化, epdlogp shape (5448, 1), discounted_epr shape (5448, 1)
            epdlogp *= discounted_epr
            # grad {'W1':dW1, 'W2':dW2}, w1和w2参数
            grad = policy_backward(eph, epdlogp, epx)
            # 梯度累积, model 是字典, 里面有 model['W1'] 和model['W2'] 2个参数
            for k in model:
                # k等于 'W1'或 'W2', 梯度累积, 先不更新参数，先累加起来，类似huggface 的 transformers的accumuate的参数
                grad_buffer[k] += grad[k]
            # 计算这一次的奖励,第一次时running_reward为None，running_reward就是reward_sum
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            if verbose:
                print('重置环境 env. 这局的奖励总共是 %f. 运行的奖励是: %f' % (reward_sum, running_reward))
            # 重置env
            observation = env.reset()
        print(f"这个batch结束，我方胜利{win}次，敌方胜利{batch_size-win}次，开始更新参数")
        # 每batch_size回合执行rmsprop参数更新，优化器,，进行一次参数更新
        for k, v in model.items():
            # 获取累积的梯度, k等于 'W1'或 'W2', v是对应的参数
            g = grad_buffer[k]  # gradient
            # 利用优化器更新和梯度更新模型参数
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            #更新参数
            model[k] -= learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            # 重置批次梯度缓冲区
            grad_buffer[k] = np.zeros_like(v)
        if epoch % 2 == 0:
            print(f"第{e}个epoch结束，完成了{batch_size}局，保存模型")
            pickle.dump(model, open(save_file, 'wb'))

def do_test(verbose=False):
    """
    测试一局游戏，谁先到21分，谁赢
    :return:
    """
    #一局游戏的初始为False
    done = False
    observation = env.reset()
    # 记录下当前乒乓球的分数，我方的分数, 和对方的分数
    myscore = 0
    comscore = 0
    ober_num = 0
    prev_x = None
    while not done:
        #回合结束
        #存储上一个frame的状态x
        # 记录一个小的回合打了多少次
        # 这一小的回合接收了多少次参数
        if render:
            # 显示画面
            env.render()
        ober_num += 1
        # 对观测值进行预处理，将网络输入设置为不同图像, 即环境状态state， observation是原始画面
        cur_x = observation_extract(observation)
        # 经过简单处理后的的环境状态cur_x，x是变化的画面的状态
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        # 记录一下上一个状态
        prev_x = cur_x
        aprob, h = policy_forward(x)
        # 根据计算得出的概率，判断我们要采取的行动
        action = 2 if aprob > 0.5 else 3
        observation, reward, done, info = env.step(action)
        if reward != 0:
            if reward == -1:
                # 对方+1分
                comscore += 1
            else:
                myscore += 1
            if verbose:
                print(f"第{myscore + comscore}次小回合，进行了{ober_num}次输入图像,分出了小的回合胜负, 奖励是: {reward}, 当前我方分数是{myscore},对方分数是{comscore}")
            # 重置下这个小的回合的接收图像次数
            ober_num = 0
    if myscore > comscore:
        print(f"我赢了")
        return True
    else:
        print(f"电脑赢了")
        return False


def test_batch(num):
    mywin = 0
    for n in range(num):
        result = do_test()
        if result:
            mywin +=1
    print(f"进行了{num}轮比赛，我赢了{mywin}次，电脑赢了{num-mywin}次")


if __name__ == '__main__':
    # 超参数列表
    H = 200  # 隐层神经元数
    D = 80 * 80  # input dimensionality: 80x80 grid， 模型的参数的维度
    batch_size = 6  #更新一次参数
    epoch = 4
    learning_rate = 1e-4
    gamma = 0.99  # 奖励折扣系数
    decay_rate = 0.99  # 衰减因子 RMSProp leaky sum of grad^2
    resume = True  # 从先前的checkpoint恢复？
    test = True  # 测试模式，闭市epsilon-greedy和渲染场景图画, 还是训练模式，训练模式，会更新模型参数
    save_file = 'pong_model_bolei.p'
    render = True  # 是否显示游戏的图像界面
    verbose = False #显示日志
    model = initial_model(save_file, resume)
    env = initial_env(verbose=verbose)
    if test:
        # do_test()
        test_batch(6)
    else:
        do_train(verbose)
