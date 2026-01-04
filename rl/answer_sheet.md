# 演習解答欄

## 演習 1.1.2

作成している行
agent: agent = agents.RandomAgent(env)
env: env = gym.make('EasyMaze-v0')

呼び出されている変数・メソッド

agent:
agent.act_and_train(obs, reward, done)
agent.act(obs)
agent.get_statistics()
agent.stop_episode_and_train(obs, reward, done)
agent.stop_episode()

env:
env.reset()
env.render(render_mode)
env.step(action)

## 演習 1.1.3

平均step数

train: 50.93
test: 52.83

## 演習 1.1.8

平均step数

train: 7.77
test: 5.0

train first 10: 15.5
train last 10: 8.1

## 演習 1.1.10

平均step数

train: 13.47
test: 5.0
