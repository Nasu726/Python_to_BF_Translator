# Python to BF Translator

Pythonの実用的なサブセットをBrainfuckへ変換するコンパイラです。

## 使い方

ユーザーが実行するファイルは **`main.py` だけ** です。

```bash
python main.py program.py
```

同じディレクトリに `program.bf` が生成されます。コンパイル後には生成されたBrainfuckのbyte数も表示します。

生成後の実行にPythonは不要です。`program.bf` を通常のBrainfuckインタプリタへそのまま渡してください。

```text
program.py --(Pythonでコンパイル)--> program.bf --(Brainfuck interpreter)--> 実行
```

`pybf/` はコンパイラ内部実装、`tests/` はテスト、`legacy/` は旧実装です。通常は直接実行しません。

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
- `input().split()` → Brainfuck上の空白tokenizerとnewline管理
- string/list → tape上の固定レイアウト
- listの動的index → Brainfuck上のindex比較・走査
- 一時変数・変数配置 → コンパイル時にtape addressへ割当

したがって、生成した `.bf` だけを別PCへ持っていっても、対応するBrainfuckインタプリタだけで実行できます。

現在の実行モデルは以下を前提とします。

- **実行開始時、使用可能なtape cellはすべて0で初期化されている**
- 8-bit wrapping cells (`255 + 1 == 0`)
- cell 0より左へは移動しない
- 必要量を確保できる右方向のtape
- `,` / `.` によるbyte I/O

zero-initialized tapeは生成コード最適化の正式なABI前提です。コンパイラは、まだ一度も書き込まれていないセルへの不要な`[-]`等を省略できます。

`pybf/bf_runtime.py` はCIで生成物を検証するためのテスト用Brainfuckインタプリタであり、生成された `.bf` の依存ランタイムではありません。

## 生成Brainfuckのサイズ

競技プログラミングでは提出ソースサイズにも上限があるため、生成物の小ささをcorrectnessと並ぶ実用要件として扱います。

現在のpublic compilerでは、標準Brainfuckの意味を保ったまま以下を行います。

- zero-initialized tapeを利用した不要clear / dead loop除去
- 隣接するpointer移動・cell加減算の正規化
- 頻繁に使うscratch cellをデータ近傍へ配置して`>` / `<`を削減
- `for c in s`の1文字loop変数を、安全な場合はcompactな1-byte payloadとして保持
- `c == "A"`等の1文字比較をbyte比較へspecialize
- 安全性を証明できる` s = input(); for c in s:`パターンでは、255-byte文字列のmaterializeと再走査をせず入力を直接loopへstreamingする

streaming optimizationは、loop body内に別の`input()`がある場合や、元のstring値が後から必要になる場合には適用しません。`break`時も元の`input()`と同じく現在行を最後まで消費してから次へ進みます。

`main.py` は生成byte数を表示し、512 KiBを超えた場合には警告します。CIにも代表的な短いstring走査について512 KiB以下を要求するregression testがあります。

## 固定ランタイム型

CLIオプションで型サイズは変更できません。型表現はABIとして固定しています。

- `int`: signed 64-bit two's complement
- `bool`: 64-bit scalar上の0/1
- `str`: 最大255 byteの固定長領域（NUL終端）
- `list[int]`: 最大64要素、各要素signed 64-bit
- `list[str]`: 最大64要素、各要素は固定長byte string

Python本来の任意精度int、動的長string/list、完全なobject alias semanticsとは異なります。

## 入力

通常の競技プログラミング向けPythonと同じ書き方を使えます。

```python
n = int(input())
a, b = map(int, input().split())
name, country = input().split()
A = list(map(int, input().split()))
words = input().split()
```

各 `input()` はBrainfuck実行時にも **1行単位** です。整数化・空白tokenize・string/listへの格納・newline管理はすべて生成Brainfuck自身が行います。

`int(input())` は最初の整数tokenを読み、同じ物理行に残りがあってもその行を最後まで消費してから次の `input()` へ進みます。`input().split()` 系もnewlineを越えて次行のtokenを盗みません。

固定長ABIのため、list容量を超えるtokenは保存せず同じ行の残りをdrainし、次の`input()`が次行から始まるようにします。現時点ではPythonの`ValueError`等の例外再現までは行いません。固定数unpackで値が不足した場合は不足分を0または空文字列、余剰分は同じ行内で破棄する固定runtime仕様です。

整数入力の実例は `examples/input_patterns.py` にあります。

## 主な対応構文

- 整数・bool・文字列・`list[int]`・`list[str]`リテラル
- `+ - * // % **`
- `& | ^ ~ << >>`
- `== != < <= > >=`
- `and / or / not`
- `if / else`, `while`, `for ... in range(...)`
- `for x in list`, `for c in string`
- `break`, `continue`, loop `else`
- `input()`, `int(input())`
- `a, b = map(int, input().split())`
- `a, b = input().split()`
- `a, b = map(str, input().split())`
- `A = list(map(int, input().split()))`
- `S = input().split()` / `list(input().split())`
- `S = list(map(str, input().split()))`
- int/string listのindex、代入、`append`, `len`, iteration
- runtime list repetition (`[x] * n`, `A * n`, `n * A`) ※現行固定容量まで
- runtime負index
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

CIでは生成したBrainfuckを実際にインタプリタで実行して検証します。また、公開APIテストで生成物が標準Brainfuck 8命令だけであることを検査します。生成サイズについても、競プロ提出を想定したregression gateを段階的に追加します。
