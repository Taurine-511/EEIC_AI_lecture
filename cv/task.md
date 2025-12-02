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
```bash
GPU: 0
# unit: 3
# Minibatch-size: 100
# epoch: 20

[epoch 1] loss: 1212.474
[epoch 2] loss: 1100.539
[epoch 3] loss: 954.826
[epoch 4] loss: 839.878
[epoch 5] loss: 770.063
[epoch 6] loss: 728.612
[epoch 7] loss: 700.602
[epoch 8] loss: 680.339
[epoch 9] loss: 662.116
[epoch 10] loss: 646.173
[epoch 11] loss: 633.079
[epoch 12] loss: 623.297
[epoch 13] loss: 614.716
[epoch 14] loss: 605.700
[epoch 15] loss: 595.175
[epoch 16] loss: 581.710
[epoch 17] loss: 568.408
[epoch 18] loss: 556.935
[epoch 19] loss: 547.712
[epoch 20] loss: 538.405
Finished Training

real	2m52.457s
user	4m36.277s
sys	0m20.599s
```

```bash
GPU: 0
# unit: 10000
# Minibatch-size: 100
# epoch: 20

[epoch 1] loss: 653.621
[epoch 2] loss: 243.725
[epoch 3] loss: 190.458
[epoch 4] loss: 167.693
[epoch 5] loss: 153.212
[epoch 6] loss: 142.712
[epoch 7] loss: 133.747
[epoch 8] loss: 126.269
[epoch 9] loss: 119.621
[epoch 10] loss: 113.598
[epoch 11] loss: 108.176
[epoch 12] loss: 103.012
[epoch 13] loss: 98.420
[epoch 14] loss: 94.093
[epoch 15] loss: 89.918
[epoch 16] loss: 86.238
[epoch 17] loss: 82.596
[epoch 18] loss: 79.425
[epoch 19] loss: 76.283
[epoch 20] loss: 73.245
Finished Training

real	8m15.607s
user	9m12.609s
sys	0m29.341s
```

### 演習 4.2.6 下の図 4.10 のネットワークを MNIST CNN クラスに実装せよ.ただし,カーネルサイズはどちらの畳み込み層についても (5 × 5) とする.MNIST CNN クラスを実装したら,train mnist cnn.py 中の”net = MLP(unit, 28*28, 10)”を書き換えて実装した MNIST CNNクラスが呼び出されるように変更し,同様に実行することで精度を確かめよ.興味がある人は,test mnist mlp.py を参考にテスト用のコードを書き実行せよ.
resultにある


### 演習 4.2.7 train cifar10.py の MyCifarDataset クラスを実装することで,cifar10 の画像を読み込むコードを完成させよ.MyCifarDataset クラスは dataset.py のなかで定義しているので,その穴埋めをせよ.CIFAR-10 の画像は mini cifar/train/にある.
### 演習 4.2.8 出力された model から最も精度が良いモデルを選び,その model を用いて,test cifar10.py を実行せよ.以下のコマンドで test cifar10.py を実行できる.
```bash
GPU: -1
# Minibatch-size: 100
# epoch: 20

[epoch 1] loss: 32.253
[epoch 2] loss: 32.251
[epoch 3] loss: 32.232
[epoch 4] loss: 32.218
[epoch 5] loss: 32.216
[epoch 6] loss: 32.203
[epoch 7] loss: 32.191
[epoch 8] loss: 32.184
[epoch 9] loss: 32.176
[epoch 10] loss: 32.169
[epoch 11] loss: 32.151
[epoch 12] loss: 32.148
[epoch 13] loss: 32.135
[epoch 14] loss: 32.114
[epoch 15] loss: 32.105
[epoch 16] loss: 32.081
[epoch 17] loss: 32.059
[epoch 18] loss: 32.048
[epoch 19] loss: 32.025
[epoch 20] loss: 32.005
Finished Training
```

```bash
Accuracy of airplane : 50 %
Accuracy of automobile :  0 %
Accuracy of  bird :  0 %
Accuracy of   cat :  0 %
Accuracy of  deer :  0 %
Accuracy of   dog : 10 %
Accuracy of  frog :  0 %
Accuracy of horse :  0 %
Accuracy of  ship : 75 %
Accuracy of truck :  0 %
Accuracy : 13.500 %
```


### 演習 4.2.9 (発展課題) 本ネットワークの精度を上げてみよ.例えば,ネットワーク構造や,最適化手法などを変化させて見ると良い.
normalizationを適用
```bash
GPU: -1
# Minibatch-size: 100
# epoch: 20

[epoch 1] loss: 32.428
[epoch 2] loss: 32.163
[epoch 3] loss: 31.963
[epoch 4] loss: 31.745
[epoch 5] loss: 31.512
[epoch 6] loss: 31.253
[epoch 7] loss: 30.901
[epoch 8] loss: 30.477
[epoch 9] loss: 30.012
[epoch 10] loss: 29.389
[epoch 11] loss: 28.968
[epoch 12] loss: 28.600
[epoch 13] loss: 28.224
[epoch 14] loss: 27.952
[epoch 15] loss: 27.680
[epoch 16] loss: 27.410
[epoch 17] loss: 27.115
[epoch 18] loss: 26.647
[epoch 19] loss: 26.311
[epoch 20] loss: 26.097
Finished Training
```

AdamWに変更
```bash
GPU: -1
# Minibatch-size: 100
# epoch: 20

[epoch 1] loss: 31.802
[epoch 2] loss: 29.267
[epoch 3] loss: 27.100
[epoch 4] loss: 25.554
[epoch 5] loss: 24.646
[epoch 6] loss: 23.433
[epoch 7] loss: 22.662
[epoch 8] loss: 22.176
[epoch 9] loss: 21.351
[epoch 10] loss: 20.799
[epoch 11] loss: 20.598
[epoch 12] loss: 19.897
[epoch 13] loss: 19.345
[epoch 14] loss: 18.911
[epoch 15] loss: 18.003
[epoch 16] loss: 17.884
[epoch 17] loss: 17.414
[epoch 18] loss: 16.936
[epoch 19] loss: 16.529
[epoch 20] loss: 16.150
Finished Training
```

siluに変更
```bash
GPU: -1
# Minibatch-size: 100
# epoch: 20

[epoch 1] loss: 31.550
[epoch 2] loss: 28.672
[epoch 3] loss: 26.531
[epoch 4] loss: 25.228
[epoch 5] loss: 23.978
[epoch 6] loss: 23.065
[epoch 7] loss: 22.250
[epoch 8] loss: 21.377
[epoch 9] loss: 20.462
[epoch 10] loss: 19.741
[epoch 11] loss: 18.912
[epoch 12] loss: 18.572
[epoch 13] loss: 17.671
[epoch 14] loss: 17.227
[epoch 15] loss: 16.661
[epoch 16] loss: 16.269
[epoch 17] loss: 15.732
[epoch 18] loss: 15.422
[epoch 19] loss: 14.885
[epoch 20] loss: 14.190
Finished Training
```

最終スコア
``bash
GPU: -1
# Minibatch-size: 100

Accuracy of airplane : 50 %
Accuracy of automobile : 80 %
Accuracy of  bird : 20 %
Accuracy of   cat : 35 %
Accuracy of  deer : 20 %
Accuracy of   dog : 50 %
Accuracy of  frog : 45 %
Accuracy of horse : 45 %
Accuracy of  ship : 45 %
Accuracy of truck : 55 %
Accuracy : 44.500 %
```



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

