# RL課題

### 演習 3.4.1 ソースコードをダウンロードし,main.py を実行せよ.また,README.md を読み,random_agent.py と easymaze_env.py の場所を確認せよ.

### 演習 3.4.2 main.py の中で,env および agent を作成している行を答えよ.また,これらに対して呼び出されているインスタンス変数・メソッドをすべて挙げよ.自由課題などで env やagent を自分で作成する場合,最低限これらの変数・メソッドをすべて実装する必要がある(インタフェース).

```python
agent = agents.RandomAgent(env)

# メソッド
agent.act_and_train(obs, reward, done)
agent.act(obs)
agent.get_statistics()
agent.stop_episode_and_train(obs, reward, done)
agent.stop_episode()
```

```python
env = gym.make('EasyMaze-v0')

# インスタンス変数
env.metadata

# メソッド
env.reset()
env.render(render_mode)
env.step(action)
```

### 演習 3.4.3 train, test のそれぞれで 100 episodes の実験を行って,ランダムエージェントがゴールするまでの episode ごとの平均 step 数を出力せよ.main.py の編集が必要となる.300 steps 掛かってもゴールできない場合,300 steps でゴールしたものとして平均を計算せよ.

```bash
Train Average Steps: 50.93
Test Average Steps: 52.83
最初の10件の平均ステップ: 52.8
最後の10件の平均ステップ: 75.3
```

### 演習 3.4.4 コメントを参考に,穴あきファイル rulebase_agent.py の穴を埋めることで,この形状の迷路を最短 step 数で解くルールベースエージェントを作成せよ.「壁にぶつかるまで下に行き,ぶつかったら右に行く」ように実装すればよい.エージェントが受け取る観測を,print() 関数を用いたり easymaze_env.py を読んだりすることで確認することが必要となることに注意せよ.

ステージは固定なので、下に2回進んでから、右に3回進めば良い
```bash
Train Average Steps: 5.00
Test Average Steps: 5.00
最初の10件の平均ステップ: 5.0
最後の10件の平均ステップ: 5.0
```

### 演習 3.4.5 easymaze_env.py を編集し,迷路に最小限の変更を行うことで,演習 3.4.4 で作成したエージェントがゴールできないような環境にせよ.ただし,別のルートでゴールできるようにしておくこと.
```bash
Train Average Steps: 300.0
Test Average Steps: 300.0
最初の10件の平均ステップ: 300.0
最後の10件の平均ステップ: 300.0
```

### 演習 3.4.6 コメントを参考に,穴あきファイル table_q_agent.py の穴を埋めることで,table-Q エージェントを実装せよ.ルールベースエージェントの代わりに table-Q エージェントを用いるために,main.py も変更する必要があることに注意せよ.
### 演習 3.4.7 演習 3.4.3 と同様に,100 episodes の実験を行って,table-Q エージェントがゴールするまでの平均 step 数を出力せよ.また,train のうち,最初の 10 episodes,最後の 10 episodesの平均 step 数を出力せよ.
```bash
Train Average Steps: 7.77
Test Average Steps: 5.0
最初の10件の平均ステップ: 15.5
最後の10件の平均ステップ: 8.1
```

### 演習 3.4.8 table-Q エージェントには,迷路の各地点における Q 値を文字列形式で出力するq_table_to_str() メソッドが用意されている.このメソッドを適宜呼び出し print() することで,Q 値が学習されていく過程を観察せよ.時間があれば,割引率や学習率,迷路の報酬などを変えて,学習の過程がどのように変わるかを観察してみよ.

### 演習 3.4.9 Table-Q エージェントを,train 時に ε-greedy で行動し,test 時に greedy で行動するようにせよ.その後,演習 3.4.3 と同様に,100 episodes の実験を行って,エージェントがゴールするまでの平均 step 数を出力し,train 時と test 時の結果を比較せよ.
testではgreedyに行動するため、最終的には最適な行動が必ず選択できるようになっている。
```bash
train episode: 100 T: 8.05 R: 10.0 statistics: []
test episode: 100 T: 5.0 R: 10.0 statistics: []
```

### 演習 3.4.10 main.py を編集し,DQN エージェント (agents/dqn_agent.py) にこの迷路を解かせてみよ.時間があれば,DQN のモデルを色々変えて(線形回帰にしてみる,活性化関数を変えてみるなど),学習がどのように変わるかを観察せよ.DQN のモデルは,agents/models/dqn_model.py にある.
default
```bash
Train Average Steps: 23.73
Test Average Steps: 5.0
最初の10件の平均ステップ: 28.9
最後の10件の平均ステップ: 12.7
```

change relu → silu
```bash
Train Average Steps: 16.41
Test Average Steps: 5.0
最初の10件の平均ステップ: 71.5
最後の10件の平均ステップ: 6.9
```

num layer 2 → 3
```bash
Train Average Steps: 20.34
Test Average Steps: 5.0
最初の10件の平均ステップ: 73.7
最後の10件の平均ステップ: 6.9
```

num layer 2 → 3 + change relu → silu
```bash
Train Average Steps: 13.47
Test Average Steps: 5.0
最初の10件の平均ステップ: 33.0
最後の10件の平均ステップ: 9.2
```

### 演習 3.4.11 CartPole に対して DQN を学習させ,様子を観察せよ.もし余裕があれば,ローカル環境(学科 PC など)でも試してみるとよい.ローカル環境では,main.py の prints_detail変数を True にすることで,エージェントが CartPole をプレイする様子を動画で見ることができる.ゲームは棒が 15 度傾いた時点で終わってしまうので,各 episode が一瞬で終わってしまって動画がよくわからないかもしれない.その場合は,done が True になってもループを抜けずに env.render() を続けるようにするとよい.
```bash
Train Average Steps: 12.95
Test Average Steps: 9.35
最初の10件の平均ステップ: 12.1
最後の10件の平均ステップ: 17.3
```


### 演習 3.4.12 CartPole において,なるべく高得点を取れるようなエージェントを作成せよ。現在の DQN のどこに問題があるのかを考え・調べ,様々な方法を試してみよ.

現状は入力空間が広すぎるため、正規化してやる必要がある。
訓練・テストの段階で、遭遇しうる現実的なレンジで見る必要がある。
Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)

同じくrewardも正規化してやる必要がある。

train episode 100 → 200
```bash
Train Average Steps: 70.945
Test Average Steps: 108.97
最初の10件の平均ステップ: 13.3
最後の10件の平均ステップ: 111.8
```

lr 0.01 → 0.005 + train episode 100 → 200
```bash
Train Average Steps: 72.26
Test Average Steps: 128.66
最初の10件の平均ステップ: 10.6
最後の10件の平均ステップ: 118.4
```


