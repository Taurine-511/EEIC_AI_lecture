# CV演習

### 演習 4.2.1 train mnist mlp.py を GPU で実行せよ.
resultフォルダに結果が格納されている

### 演習 4.2.2 白黒の手書き数字画像を作成せよ.この時黒地に白文字で数字を書くことに注意する.(学習させた手書き数字データが黒地に白なので) 時間がなければ,”cv/image.png”もしくは”cv/data/mini mnist”中の画像を使用して 4.2.3 を実行する.
やってない

### 演習 4.2.3 自分が書いた手書き文字を先程学習したモデルに通して出力を表示し,正しく数字が認識されたか確かめよ.
すでに用意されているimage.png(5)は8と分類された。

### 演習 4.2.4 test mnist mlp.py を書き換えることで予測数字を確信度が高い方から上位3つまで出すように変更せよ.
maxからtopkに変更

### 演習 4.2.5 MLP の中間ノード数を1. 3 / 2. 10000にしてそれぞれ 20epoch 学習し,最終的な精度と学習にかかった時間を比較せよ.
colabで実施の必要あり

### 演習 4.2.6 下の図 4.10 のネットワークを MNIST CNN クラスに実装せよ.ただし,カーネルサイズはどちらの畳み込み層についても (5 × 5) とする.MNIST CNN クラスを実装したら,train mnist cnn.py 中の”net = MLP(unit, 28*28, 10)”を書き換えて実装した MNIST CNNクラスが呼び出されるように変更し,同様に実行することで精度を確かめよ.興味がある人は,test mnist mlp.py を参考にテスト用のコードを書き実行せよ.
resultにある


### 演習 4.2.7 train cifar10.py の MyCifarDataset クラスを実装することで,cifar10 の画像を読み込むコードを完成させよ.MyCifarDataset クラスは dataset.py のなかで定義しているので,その穴埋めをせよ.CIFAR-10 の画像は mini cifar/train/にある.
### 演習 4.2.8 出力された model から最も精度が良いモデルを選び,その model を用いて,test cifar10.py を実行せよ.以下のコマンドで test cifar10.py を実行できる.


### 演習 4.2.9 (発展課題) 本ネットワークの精度を上げてみよ.例えば,ネットワーク構造や,最適化手法などを変化させて見ると良い.


### 演習 4.3.1 create db.py で mini cifar/train/のデータベースを作り, mini cifar/test/から適当に画像を選んで search.py を CPU 上で実行せよ. (ソースコードの argparse 部分をよく読むこと)
検索した通りのものが最上位に出た
```bash
data/mini_cifar/train/./airplane/airbus_s_000129.png
data/mini_cifar/train/./airplane/jetliner_s_000400.png
data/mini_cifar/train/./airplane/airbus_s_000290.png
data/mini_cifar/train/./ship/containership_s_000128.png
data/mini_cifar/train/./ship/hospital_ship_s_002012.png
```

### 演習 4.3.2 search.py の中で深層特徴 (src df) を計算する部分を探し,深層特徴の shape を表示せよ.
src_features.shapeを表示
```bash
Shape: torch.Size([1, 4096])
```

### 演習 4.3.3 create db.py の中で深層特徴として使用する隠れ層を指定している部分を探し,VGG16 の fc6 層 (もっとも浅い fc 層) を利用するよう書き換えよ.その後もう一度 create db.pyと search.py を実行せよ.
network_db.pyのforward()内のself.sequencialsを線形層を一つ経過した段階で出力するように変更

検索した通りのものが最上位に出て、airplaneが検索候補に増えた。
```bash
data/mini_cifar/train/./airplane/airbus_s_000129.png
data/mini_cifar/train/./airplane/airbus_s_000290.png
data/mini_cifar/train/./airplane/jetliner_s_000400.png
data/mini_cifar/train/./airplane/jumbojet_s_000520.png
data/mini_cifar/train/./ship/boat_s_001839.png
```

### 演習 4.3.4 MNIST で MLP を 訓 練 し ,そ の 隠 れ 層 出 力 を 深 層 特 徴 と し て 利 用 し てmini mnist/train/のデータベースを作った後, mini mnist/test/から適当に選んだ画像と訓練した MLP を用いて search.py を実行せよ. 
```bash
python search.py --input data/mini_mnist/train/6/204.png
Shape: torch.Size([1, 1000])
data/mini_mnist/train/./6/204.png
data/mini_mnist/train/./6/13.png
data/mini_mnist/train/./6/147.png
data/mini_mnist/train/./6/155.png
data/mini_mnist/train/./6/83.png
```

