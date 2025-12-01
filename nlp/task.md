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
