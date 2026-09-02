# Python to BF Translator

Pythonの実用的なサブセットをBrainfuckへ変換するコンパイラです。

## 使い方

ユーザーが実行するファイルは **`main.py` だけ** です。

```bash
python main.py program.py
```

同じディレクトリに `program.bf` が生成されます。

`pybf/` はコンパイラ内部実装、`tests/` はテスト、`legacy/` は旧実装です。通常は直接実行しません。

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
pybf/            # コンパイラ・Brainfuck runtime・演算実装
examples/        # 入力Pythonの例
tests/           # pytest / CI
legacy/          # 旧実装・参考コード
```

## 開発者向け

```bash
python -m pytest -q
```

CIでは生成したBrainfuckを実際にインタプリタで実行して検証します。
