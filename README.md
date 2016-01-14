# Tensordot
Code generator for tensor contraction

## 使用方法

全探索の場合
```
./tdt.py input_file
```

ランダム探索の場合
```
./rtdt.py input_file
```

## 入力ファイルの書式

入力ファイルは1行につき1つのテンソルを，以下の形式で定義します．
```
tensor  テンソル名  ボンド名  ボンド名 …
```

次のような行を加えることで，ボンド次元が設定可能です．ボンド次元のデフォルト値は10です．
```
bond  ボンド名  次元数
```

その他のオプションについては，`example/input.dat`を参照して下さい．
