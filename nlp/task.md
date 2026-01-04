# NLP課題

### 演習 2.7.1 適当な単語または単語列を入力し,入力によって出力が変化すること,ある程度「日本語らしい」文章になることを確認せよ.
```bash
こんにちは
、
トム
。
```

### language_model_lstm.py における LanguageModel クラスの forward() メソッドを実装し,実際に学習を行ってみよ.学習は language_model_lstm_train.py を実行することで行うことができる.学習が終了したら,language_model_lstm_test.py を実行し,動作を確認せよ.
```bash
こんにちは
、
その
島
の
中
で
は
その
島
の
中
で
は
その
島
の
中
で
は
その
島
の
中
で
は
その
島
の
中
で
```

```bash
ライン
川
が
好き
です
。
```


### language_model_lstm_train.py 及び language_model_lstm_test.py を書き換え,英語の言語モデルを学習し,動作を確認せよ.学習には時間がかかるため,学習中に次の章を読み進めておくと良い.
```bash
i
have
a
lot
of
the
world
.
```

### 演習 2.7.4 translator_model_train.py を実行し,翻訳モデルの学習を行ってみよ.学習している間に,translator_model.py 及び translator_model_train.py を読んで,ネットワークを定義している部分,エンコードやデコードを行う部分などを探してみよ.

### 演習 2.7.5 学習が終了したら,translator_model_test.py を実行し,結果を確かめてみよ.入力の際は「I have a book .」のように,ピリオドも単語として区切るため,スペースを挟む必要があることに注意する.
```bash
I have a book .
私は君の助力が必要だ。

hello world .
悪貨は良貨を駆逐する。
```

### 演習 2.7.6 translator_model_test.py を書き換え,translator_full.model による翻訳を試してみよ.
```bash
I have a book .
私はその本を読むべきだった。

hello world .
概して、政府は人にされている。
```

### 演習 2.7.7 sentence_data.py の__init__ で学習データを読み込んでいる部分を書き換え,データセットに「<UNKNOWN>」が含まれるようにせよ.

### 演習 2.7.8 language_model_lstm_test.py や,translator_model_test.py を書き換え,未知の単語が入力された場合に「<UNKNOWN>」に置き換えて,出力が得られるようにせよ.これをテストする場合は,前の課題で書き換えた sentence_data.py を用いて学習をやり直す必要があることに注意せよ.
```bash
コンんち hello world .  
その島の住民は友好的だ。
```