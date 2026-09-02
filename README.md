# Python to BF Translator

AtCoder等で動かせるBrainfuckコードへ、実用的なPythonサブセットを変換することを目指しています。
整数は固定64bit two's-complementです。

> `main` の旧実装は残しています。現在の大規模改修は `gpt56-v2-core` ブランチ / PR #1 で開発中です。

## マージ前に試す

```bash
git fetch origin
git switch gpt56-v2-core
python -m pytest -q
```

PythonファイルをBrainfuckへ変換:

```bash
python transpiler_v3.py examples/demo_v3.py -o demo.bf
```

変換してそのまま内蔵Brainfuck runtimeで実行:

```bash
python try_transpiler.py examples/demo_v3.py --run
```

入力をファイルから与える場合:

```bash
printf "nasu\n7\n12\n" > input.txt
python try_transpiler.py examples/demo_v3.py --run --input-file input.txt
```

Brainfuckだけ出力するなら:

```bash
python try_transpiler.py examples/demo_v3.py > demo.bf
```

文字列変数の固定容量は既定128 byteです。変更する場合:

```bash
python try_transpiler.py program.py --run --string-capacity 255
```

## 現在のv3で扱えるもの

### 整数 / bool

- 64bit signed two's-complement整数
- `+ - * // %`
- 定数指数の `**`
- `& | ^ ~`
- 定数shift `<< >>` (`>>` は符号拡張)
- `== != < <= > >=`
- chained comparison (`a < b < c`) と短絡評価
- `and / or / not`（v2のscalar値についてPython式の短絡評価）
- `abs(x)`, `bool(x)`, `min(a,b)`, `max(a,b)`
- `int(input())`

### 文字列

v3ではNUL終端の固定容量byte列として文字列変数を持てます。

```python
name = input()
copy = name
print("hello", copy)
print(len(copy))
print(copy == "nasu")
```

- 文字列リテラル代入
- `input()`
- 文字列変数コピー
- `print(str)`
- `len(str)`
- `==`, `!=`
- 長すぎる入力行は容量まで保存し、残りをnewlineまで読み捨てるため次の`input()`を壊さない

現在は文字列連結・slice・動的index・Unicode object semanticsは未実装です。

### 文 / 制御構造

- 単純代入
- `a = b = expr`
- tuple/list unpacking (`a, b = b, a` を含む)
- `+= -= *= //= %= &= |= ^=`
- `if / else`
- `while`
- `for i in range(...)`（stepは現在定数）
- `print(...)`
- `print(..., sep="...", end="...")`
- ternary expression `x if cond else y`（scalar）

## メモリ管理

v2/v3では、AST visitorから直接適当なセル番号を振る方式をやめています。

- 演算workspaceを共有
- expression temporaryはregion allocatorで文ごとにmark/rewindして再利用
- v3の変数領域はlive-rangeを解析して、寿命が重ならない固定長blockを再利用
- loop back-edgeを壊さないよう、ループ内で現れる変数は保守的にpinして共有禁止
- int64とstringを異なるblock sizeとして同じstatic allocatorで配置

今後、List等を入れる段階ではこの上にruntime heap / indexed storageを追加します。

## backend

主な実装:

- `bfcore.py`: pointer追跡・copy・64bit ADD/SUB/EQ
- `bfwordops.py`: bitwise / shift
- `bfcompare.py`: signed/unsigned compare
- `bfarith.py`: MUL / unsigned DIVMOD
- `bfsigned.py`: Python仕様のsigned `//` / `%`
- `bfio.py`: decimal int input/output
- `bfstrings.py`: 固定容量string runtime
- `bfmemory.py`: static live-range allocator
- `bfquad.py`: `[marker, bit0, bit1]` のbase-4実験backend
- `transpiler_v2.py`: scalar frontend
- `transpiler_v3.py`: typed int/string frontend
- `bf_runtime.py`: テスト/試用用BF interpreter

## これから

優先度高:

- List (`list(map(int, input().split()))`, indexing, assignment, len)
- `input().split()` / `map(int, ...)`
- `break` / `continue`
- runtime shift amount
- string concatenation / indexing / slicing
- functions（call frame設計が必要）
- base-4 backendをfrontendから選択可能にしてbinary版と実測比較

後回し:

- Set / Dict
- TrueDiv / float
- CPython互換Unicode object model

完全なPython実装ではなく、まず競技プログラミングで頻出する書き方をBrainfuck上で正しく動かすことを目標にしています。
