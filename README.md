# Tensordot

[![DOI](https://zenodo.org/badge/38998367.svg)](https://zenodo.org/badge/latestdoi/38998367)

Code generator for tensor contraction

## 使用方法

全探索の場合

```
./tdt.py input_file
```

## 入力ファイルの書式

入力ファイルは1行につき1つのテンソルを，以下の形式で定義します．

```
tensor  テンソル名  ボンド名  ボンド名 …
```

次のような行を加えることで，ボンド次元が設定可能です．ボンド次元のデフォルト値は10です．

```
bond_dim  次元数 ボンド名 …
```

その他のオプションについては，`example/input.dat`を参照して下さい．


# 参考文献

* R. N. C. Pfeifer, et al.: Phys. Rev. E 90, 033315 (2014)
