# Python to BF Translator

AtCoder等で動かせるBrainfuckコードへ、実用的なPythonサブセットを変換することを目指しています。整数は固定64bit two's-complementです。

> `main` の旧実装は残しています。現在の大規模改修は `gpt56-v2-core` ブランチ / PR #1 で開発中です。

## マージ前に試す

```bash
git fetch origin
git switch gpt56-v2-core
python -m pytest -q
```

一番簡単なのは `try_transpiler.py` です。Python→BF変換後、そのBFを内蔵runtimeでそのまま実行できます。

```bash
python try_transpiler.py examples/demo_current.py --run
```

入力をファイルから与える場合:

```bash
printf "nasu\n7 12\n1 2 3 4\n" > input.txt
python try_transpiler.py examples/demo_current.py --run --input-file input.txt
```

短い入力なら:

```bash
python try_transpiler.py sample.py --run --input-text '7 -2\n'
```

生成BFも保存して実行する:

```bash
python try_transpiler.py sample.py -o sample.bf --run --input-file input.txt
```

Brainfuckだけstdoutへ出す:

```bash
python try_transpiler.py sample.py > sample.bf
```

固定容量は変更できます。

```bash
python try_transpiler.py program.py --run --string-capacity 255 --list-capacity 64
```

`--backend v2` / `--backend v3` は古いfrontendとの回帰比較用です。通常は既定の `current` を使ってください。

## 現在扱える主なPython

### int / bool

- signed 64bit two's-complement
- `+ - * // %`
- 定数非負指数の `**`
- `& | ^ ~`
- 定数shift `<< >>` (`>>` は符号拡張)
- `== != < <= > >=`
- chained comparison (`a < b < c`) と短絡評価
- scalar `and / or / not`
- `abs(x)`, `bool(x)`, `min(a,b)`, `max(a,b)`
- `int(input())`
- `a, b = map(int, input().split())`

`//` と `%` はC風truncateではなくPythonのfloor semanticsへ合わせています。

### str

NUL終端・固定容量のbyte stringです。

```python
name = input()
copy = name
print("hello", copy)
print(len(copy))
print(copy == "nasu")
```

- string literal
- `input()`
- 代入/コピー
- `print(str)`
- `len(str)`
- `==`, `!=`
- overlong input lineは容量まで保存した後、残りをnewlineまでdrainするので次の`input()`を壊さない

現在はUnicode object model、concat、index/sliceは未実装です。

### list[int]

固定容量のint64 listを扱えます。内部表現は `[length: 1 byte][int64 ...]` です。

```python
A = [1, 2, 3]
A[1] = 7
A.append(9)
print(A, len(A), A[-1])

B = list(map(int, input().split()))
for x in B:
    print(x)
```

- int list literal
- `list(map(int, input().split()))`
- `len(A)`
- constant / runtime index read
- constant / runtime index assignment
- negative constant index (runtime length基準)
- `append`
- `for x in A`
- `print(A)` (`[1, 2, 3]`形式)
- input lineが容量を超えた場合は超過tokenをnewlineまでdrain

現在の重要な差: `B = A` はPythonの参照aliasではなくvalue copyです。また範囲外runtime indexはまだ`IndexError`を再現せず、readは0 / writeはno-opです。contestで有効なindexを使う前提です。

### 文 / 制御構造

- assignment / chained assignment
- tuple/list unpack (`a, b = b, a`)
- `+= -= *= //= %= &= |= ^=`
- `if / else`
- `while`
- `for i in range(...)`（stepは現在定数）
- `for x in int_list`
- `break`
- `continue`
- `for ... else` / `while ... else`
- `print(..., sep="...", end="...")`
- scalar ternary expression `x if cond else y`

Brainfuckにはbreak命令がないため、`break/continue` はiterationごとのbody-active flagとbreak flagでloweringしています。

## メモリ管理

旧v1の「visitorが直接セル番地を決めてBFを書く」方式から分離しました。

- 演算workspaceを共有
- expression temporaryはregion allocatorで文ごとにmark/rewind
- static variableはlive-rangeを解析して寿命が重ならないblockを再利用
- loop back-edgeで誤再利用しないよう、loop内に現れる変数は保守的にpin
- int / string / listの異なるblock sizeを同じallocatorでbest-fit配置
- list lengthは1 byteに圧縮し、elementだけ64bit

将来、Pythonのlist alias / object identity / function local等へ進む場合は、この上にobject/heap layerを追加します。

## 実装レイヤ

- `bfcore.py`: pointer tracking / copy / int64 ADD/SUB/EQ
- `bfwordops.py`: bitwise / shift
- `bfcompare.py`: signed/unsigned compare
- `bfarith.py`: MUL / unsigned DIVMOD
- `bfsigned.py`: Python-style signed `//` / `%`
- `bfio.py`: decimal int I/O
- `bfstrings.py`: fixed-capacity string runtime
- `bftokens.py`: whitespace/line-aware signed integer token reader
- `bflists.py`: fixed-capacity int64 list runtime
- `bfmemory.py`: conservative static live-range allocator
- `bfquad.py`: experimental base-4 `[marker,b0,b1]` ADD/SUB backend
- `transpiler_v2.py`: scalar frontend
- `transpiler_v3.py`: int/string typed frontend
- `transpiler.py`: list / common contest syntax layer
- `transpiler_full.py`: break/continue/loop-else layer
- `try_transpiler.py`: recommended compile-and-run developer CLI
- `bf_runtime.py`: optimized test/developer BF interpreter

## 次に強化したいもの

- string concat / index / slice
- list alias semantics / list comparison / more methods
- runtime shift amount
- user functions（まずsimple function inline、その後call-frame）
- nested lists / tuple values
- base-4 backendをfrontendから選択可能にしてbinary版と実測比較
- generated BF optimizer / strength reduction

後回し:

- Set / Dict
- TrueDiv / float
- full Unicode / CPython object compatibility

完全なCPython再実装ではなく、まず「競技プログラミングで自然に書くPython」をBrainfuck上で正しく動かし、その後Python semanticsとの差を順に縮める方針です。
