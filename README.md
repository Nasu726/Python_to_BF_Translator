# Python to BF Translator

Pythonの実用的なサブセットをBrainfuckへ変換するコンパイラです。

## 使い方

ユーザーが実行するファイルは **`main.py` だけ** です。

```bash
python main.py program.py
```

同じディレクトリに `program.bf` が生成されます。

生成後の実行にPythonは不要です。`program.bf` を通常のBrainfuckインタプリタへそのまま渡してください。

```text
program.py --(Pythonでコンパイル)--> program.bf --(Brainfuck interpreter)--> 実行
```

`pybf/` はコンパイラ内部実装、`tests/` はテスト、`legacy/` は旧実装です。通常は直接実行しません。

## 入力

通常の競技プログラミング向けPythonと同じ書き方を使えます。

```python
n = int(input())
a, b = map(int, input().split())
A = list(map(int, input().split()))
```

各 `input()` はBrainfuck実行時にも**1行単位**です。整数化・空白区切りtokenize・listへの格納・次行との境界管理はすべて生成Brainfuck自身が行います。

例えば入力が

```text
3
10 20
4 5 6
```

なら上のコードでは `n == 3`, `a == 10`, `b == 20`, `A == [4, 5, 6]` になります。

固定ABIのため、`list[int]` は最大64要素です。正常なPython/競プロ入力ではtoken数が代入先と一致することを前提にします。現時点では不正なunpack個数に対するPythonの `ValueError` までは再現せず、不足分は0、余剰分は同じ行内で破棄します。

実例は `examples/input_patterns.py` にあります。

## Standalone Brainfuck contract

生成物は標準Brainfuckの8命令だけから構成されます。

```text
> < + - . , [ ]
```

Python側が実行時に計算・管理する仕組みはありません。Pythonの高水準機能はコンパイル時に、Brainfuck上で動く処理へloweringされます。

- 64-bit整数演算 → Brainfuck上のbit演算・carry/borrow処理
- 比較・条件分岐 → Brainfuckセルとloopによる制御
- `while` / `for` / `break` / `continue` → Brainfuck loopと明示的control flag
- `print(int)` → Brainfuck上のdecimal conversion
- `input()` / `int(input())` → `,` とBrainfuck上のparser
- string/list → tape上の固定レイアウト
- listの動的index → Brainfuck上のindex比較・走査
- 一時変数・変数配置 → コンパイル時にtape addressへ割当

したがって、生成した `.bf` だけを別PCへ持っていっても、対応するBrainfuckインタプリタだけで実行できます。

現在の実行モデルは以下を前提とします。

- 8-bit wrapping cells (`255 + 1 == 0`)
- cell 0より左へは移動しない
- 必要量を確保できる右方向のtape
- `,` / `.` によるbyte I/O

`pybf/bf_runtime.py` はCIで生成物を検証するためのテスト用Brainfuckインタプリタであり、生成された `.bf` の依存ランタイムではありません。

## 固定ランタイム型

CLIオプションで型サイズは変更できません。型表現はABIとして固定しています。

- `int`: signed 64-bit two's complement
- `bool`: 64-bit scalar上の0/1
- `str`: 最大255 byteの固定長領域（NUL終端）
- `list[int]`: 最大64要素、各要素signed 64-bit

Python本来の任意精度int、動的長string/list、完全なobject alias semanticsとは異なります。

## 主な対応構文

- 整数・bool・文字列・整数listリテラル
- `+ - * // % **`
- `& | ^ ~ << >>`
- `== != < <= > >=`
- `and / or / not`
- `if / else`, `while`, `for range(...)`
- `break`, `continue`, loop `else`
- `input()`, `int(input())`
- `a, b = map(int, input().split())`
- `A = list(map(int, input().split()))`
- `A[i]`, `A[i] = x`, `A.append(x)`, `len(A)`, `for x in A`
- `print(...)`, `sep=`, `end=`
- `abs`, `bool`, `min`, `max`

## リポジトリ構成

```text
main.py          # ← ユーザーが実行する唯一の入口
pybf/            # コンパイラ内部実装
tests/           # pytest / CI
examples/        # 入力Pythonの例
legacy/          # 旧実装・参考コード
```

## 開発者向け

```bash
python -m pytest -q
```

CIでは生成したBrainfuckを実際にインタプリタで実行して検証します。また、公開APIテストで生成物が標準Brainfuck 8命令だけであることを検査します。
