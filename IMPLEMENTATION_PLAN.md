# Python → Brainfuck Translator 実装計画

## 0. 最終目標

このプロジェクトの目標は、**Pythonの基本的な文法・型・制御構造・関数・可変オブジェクトの意味論を、生成されたBrainfuckだけで実行できるトランスパイラ**にすることです。

Pythonはコンパイル時にのみ使用し、生成後の `.bf` は標準Brainfuckインタプリタだけで実行できる状態を維持します。

主用途としてAtCoder等の競技プログラミングを想定し、典型的なPythonコードをBrainfuck向けに書き換えず、そのまま入力として使えることを重視します。

たとえば次のようなコードは必須対応です。

```python
n = int(input())
a = [0] * n
b = a

for i in range(n):
    a[i] = i

print(b)
```

ここで `b = a` はコピーではなくPython同様の**同一list objectへの参照共有**でなければなりません。

---

## 1. スコープ方針

### 1.1 実装対象

優先するのは「別の便利構文に頼らずPythonプログラムを書くために必要な基本機能」です。

- 代入・参照・スコープ
- 基本リテラル
- 整数・真偽値・文字列
- list / tuple / dict / set の基本操作
- 算術・比較・論理・bit演算
- 添字アクセス・更新
- `if` / `while` / `for`
- `break` / `continue`
- 関数定義・呼び出し・`return`
- 再帰
- mutable objectのalias semantics
- 行単位入力・出力
- Pythonコードで普通に使う基本的な組み込み操作

### 1.2 初期段階では実装しない発展的構文

以下は、基本構文で同じ処理を書けるため優先しません。

- list / dict / set comprehension
- generator expression
- lambda
- decorator
- generator / `yield`
- async / await
- pattern matching (`match`)
- walrus operator (`:=`)
- metaclass等の高度なobject model

これらを採用しなくても、同じアルゴリズムを基本構文で記述できることを優先します。

### 1.3 標準ライブラリ

標準ライブラリ全体の再実装は目標にしません。

`collections`, `heapq`, `bisect`, `itertools`, `math` 等は別レイヤの課題とします。

ただし、言語を普通に使うために不可欠な操作はruntime primitiveまたはcompiler intrinsicとして提供します。

例:

- `input`
- `print`
- `int`
- `str`
- `bool`
- `len`
- `range`

一方、`sum`, `sorted`, `enumerate`, `zip` 等は、それらが無くても基本構文で同値処理を書けるため後回しにできます。

---

# 2. 現在の基盤

現時点で存在する主な機能:

- signed int64
- bool scalar
- fixed byte string
- `list[int]`
- `list[str]`
- 四則演算の主要部分
- signed `//`, `%`
- bit演算
- 比較
- `if`, `while`, `for range`
- `break`, `continue`, loop `else`
- `input()`
- `int(input())`
- `input().split()`
- `map(int, input().split())`
- `list(map(int, input().split()))`
- string split
- list indexing / assignment / append / iteration
- `print`
- compile-time memory placement / temporary arena
- standalone Brainfuck出力

ただし現在のlist/string runtimeは「固定容量の値」に近く、Pythonのobject semanticsとはまだ一致していません。

特に次が大きな不足です。

```python
a = [0] * n
b = a
```

この2行を正しく扱うには、list演算を個別追加するだけでは不十分で、**runtime object / reference / heap model**が必要です。

---

# 3. 最優先: runtime object model と memory model

## Phase 1 — Heap / Reference 基盤

### 目的

mutable objectを「変数領域そのもの」ではなく、runtime objectとして管理する方式へ移行します。

### 必須要件

- scalar valueとobject referenceを区別
- list等のmutable valueはheap objectとして保持
- 変数はobject handle/referenceを保持
- `b = a` は同一objectを指す
- `b[0] = x` が `a[0]` に反映される
- objectのruntime lengthを持つ
- compile-timeにlist長が分からなくても生成可能
- Brainfuck自身がheapを操作する
- Python runtimeへ依存しない

### 想定heap object header

概念的には以下の情報を持たせます。

```text
[type]
[length]
[capacity / block size]
[reference/ownership metadata]
[data ...]
```

BFでは通常のmachine pointerを直接使えないため、handleからobjectへ到達するためのmarker traversal / indexed traversal primitiveをruntimeとして定義します。

### allocator

最初は以下の順で実装します。

1. monotonic / bump allocation
2. object handleによる参照
3. reuse可能なfree-list
4. 必要であればreference counting

競プロコードではGCそのものより、loop内で一時listを繰り返し生成してもmemoryを一方的に消費しないことの方が重要です。

cycle GCは後回しにできます。

### Acceptance tests

```python
a = [1, 2]
b = a
b[0] = 9
print(a)
# [9, 2]
```

```python
a = [1]
b = a
c = b
c.append(2)
print(a, b, c)
```

```python
for i in range(100):
    a = [i]
```

最後のケースでheapが単純リークし続けないことも確認します。

---

# 4. Dynamic list

## Phase 2 — Python listの基本意味論

Heap基盤の直後に実装します。

### 必須構文

```python
a = []
a = [1, 2, 3]
a = [0] * n
a = [x] * n
```

特にruntime値によるrepeatを必須にします。

```python
n = int(input())
a = [0] * n
```

### 必須操作

- runtime length
- runtime capacity
- append
- pop
- clear
- index load/store
- negative index
- `len(a)`
- iteration
- equality
- membership (`x in a`, `x not in a`)
- list concatenation `a + b`
- list repetition `a * n`, `n * a`
- alias assignment
- explicit copy operationが必要になった場合のcopy lowering

### index semantics

```python
a[i]
a[-1]
a[i] = x
```

runtime indexに対応します。

### slice

basic indexingが安定してから以下を追加します。

```python
a[l:r]
a[l:r:s]
a[::-1]
```

slice assignmentは後段でも構いません。

### nested list

AtCoderでは2次元配列が必須です。

comprehensionを非目標にする代わりに、次が書ける必要があります。

```python
a = []
i = 0
while i < n:
    a.append([0] * m)
    i += 1
```

またalias semanticsを正しく区別します。

```python
a = [[0] * m] * n
```

これはPython同様、同じinner listへの参照をn個持つ必要があります。

---

# 5. 型システム / Value model

## Phase 3 — Compiler-side type inference

Brainfuckにはruntime型システムがないため、コンパイル時に可能な限り型を確定します。

### 基本方針

- scalarは固定width
- mutable objectはreference
- container element typeはcompile-time inferenceを優先
- Pythonの完全なdynamic typingを最初から再現しない
- ただし普通の競プロコードで自然に型推論できることを目標にする

対象:

- int
- bool
- str
- list[T]
- tuple[T...]
- dict[K,V]
- set[T]
- function signatures

### 必須検査

- incompatible assignment
- function argument type propagation
- return type propagation
- container element type propagation
- nested container type

### 将来

必要になればtagged valueへ拡張しますが、最初から全値をtagged unionにしてBF runtimeを重くしないことを優先します。

---

# 6. Tuple / unpacking / assignment semantics

## Phase 4

### 必須

```python
a, b = b, a
x, y = pair
```

- tuple literal
- tuple value
- unpack
- nested unpack
- function multiple return

```python
def f():
    return 1, 2

a, b = f()
```

starred unpackは後回しでよいです。

---

# 7. String

## Phase 5

現在のfixed byte stringを、通常のPythonコードで必要な操作まで拡張します。

### 必須

- assignment
- equality / ordering
- `len`
- index
- negative index
- slice
- concatenation
- repetition
- iteration
- `in`
- `str(int)`
- `int(str)`

### I/O model

Brainfuckの標準I/Oがbyte単位なので、まずUTF-8 byte sequenceとして扱います。

AtCoderで主に使うASCII入力についてPythonコードと同じ結果を保証します。

完全なUnicode code point semanticsは別フェーズとします。

---

# 8. Function / Scope / Call stack

## Phase 6 — user-defined function

AtCoderで普通のPythonコードを使うための最重要項目です。

### 必須構文

```python
def f(x, y):
    z = x + y
    return z

ans = f(a, b)
```

### 必須runtime

- call frame
- local variables
- parameters
- return value
- nested calls
- recursion

### recursion acceptance

```python
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)
```

生成Brainfuckだけで動くこと。

### scope

- local
- module/global
- `global` は必要になった時点で追加

`nonlocal`, closureは発展機能として後回しにできます。

### default / keyword args

通常のpositional argumentを先に完成させます。

その後:

- default arguments
- keyword arguments

を追加します。

`*args`, `**kwargs` は優先度低。

---

# 9. Control flow 完成

## Phase 7

### for

以下を普通に扱います。

```python
for i in range(n):
for i in range(l, r):
for i in range(l, r, step):
for x in a:
for ch in s:
```

`range`のstart/stop/stepはruntime値対応にします。

現在の「stepがcompile-time constant」等の制約を除去します。

### その他

- nested break/continue
- loop else
- return through nested control flow

を統一control-flow loweringで処理します。

---

# 10. Operators 完成

## Phase 8

### int

- `+ - * // % **`
- unary `+ -`
- `& | ^ ~ << >>`
- comparison

既存実装を完全に統一します。

### comparison

- chained comparison
- `in`
- `not in`

### logical

- `and`
- `or`
- `not`
- Pythonのshort-circuit
- operand value return semantics

### identity

`is` / `is not` はobject handleが導入された後なら実装可能です。

mutable objectについてidentity比較を正しく扱います。

---

# 11. Dict / Set

## Phase 9

AtCoderで頻出するため、標準ライブラリ扱いにはせず基本containerとして実装します。

## dict

最低限:

```python
d = {}
d[key] = value
x = d[key]
key in d
len(d)
for key in d:
```

### runtime

最初はlinear tableでもcorrectness上は成立しますが、実用性を考えると最終的にはhash tableを実装します。

- open addressing
- tombstone
- resize
- fixed-width hash

を候補とします。

## set

- add
- remove/discardの基本操作
- membership
- len
- iteration

setも同じhash runtimeを共有します。

---

# 12. Numeric types

## Phase 10

### int

当面signed 64-bitを仕様として維持します。

Python arbitrary precision intは非常に大きなruntimeコストを生むため、AtCoder用途ではint64を正式ABIとして扱います。

### float

Python基本機能として後段でfloat64を追加します。

- literal
- conversion
- `+ - * /`
- comparison
- print / parse

Brainfuck上でIEEE-754を完全にsoftware実装するか、独自fixed representationを使うかは実装前に決定します。

Pythonとの意味論一致を目標にするならIEEE-754 binary64を優先します。

---

# 13. Exceptions / Error model

## Phase 11

AtCoderのvalid inputでは例外が発生しないケースが多いため優先度は低めですが、現在のようなsilent zero-fill等は最終仕様にはしません。

最低限:

- division by zero
- index out of range
- invalid unpack
- invalid int conversion
- missing dict key

に対してruntime error stateを持たせます。

完全な`try/except`実装は後回しにできます。

まず「誤った値で処理継続しない」ことを保証します。

---

# 14. Memory management 完成

## Phase 12

Heap/object systemが安定した後、長時間実行に耐えるmemory managementへ進みます。

候補:

1. compile-time lifetime analysis
2. stack/region allocationできるobjectはstackへ
3. escaping mutable objectだけheapへ
4. heap free-list
5. reference counting
6. cycleを含む場合のみ簡易GC検討

重要なのは、Brainfuck側で実行可能であることです。

Python runtimeにallocation/freeを依存させません。

---

# 15. Optimization

Correctnessの後に実施します。

## source-size optimization

- repeated primitive sharing
- runtime loop化
- pointer movement削減
- base-4 / bit-pair backend活用
- constant propagation
- dead temporary elimination

## runtime optimization

- common BF arithmetic primitive改善
- traversal回数削減
- object layout locality
- list index traversal改善
- decimal I/O改善

## memory optimization

- liveness reuse
- frame reuse
- heap block reuse

各最適化には以下を測定します。

- generated BF command count
- executed BF primitive steps
- peak tape cells

---

# 16. Differential testing

すべての主要機能について、同一のPython source/inputを

1. CPython
2. Python→BF→BF interpreter

で実行し、stdoutを比較するテストを増やします。

例外未実装部分を除き、結果不一致はregressionとします。

## AtCoder compatibility corpus

実際に典型的な競プロコードパターンをfixtureとして蓄積します。

最低限:

### 入力

```python
n = int(input())
a, b = map(int, input().split())
a = list(map(int, input().split()))
s = input()
```

### 配列

```python
a = [0] * n
for i in range(n):
    a[i] = i
```

### alias

```python
a = [0] * n
b = a
b[0] = 1
assert a[0] == 1
```

### 2次元配列

```python
a = []
for i in range(h):
    a.append([0] * w)
```

### DP

```python
dp = [0] * (n + 1)
```

### graph

```python
g = []
for i in range(n):
    g.append([])

g[u].append(v)
```

### function

```python
def dfs(v):
    ...
```

これらが「Python側で書き換え無し」にcompile/runできることを各milestoneのacceptance conditionにします。

---

# 17. 実装順序

依存関係を考慮し、以下の順で進めます。

## P0 — まず実用性を決める基盤

1. runtime object handle
2. heap allocator
3. mutable alias semantics
4. dynamic list length/capacity
5. `[x] * n`
6. nested list

この段階で次を必ず通します。

```python
n = int(input())
a = [0] * n
b = a
for i in range(n):
    a[i] = i
print(b)
```

## P1 — 普通のアルゴリズム記述

7. function / return / call frame
8. recursion
9. runtime `range` step
10. string index/slice/concat/repeat
11. tuple values/unpack
12. membership operators

## P2 — 競プロcontainer

13. nested heterogeneous container type inference
14. dict
15. set

## P3 — completeness

16. float64
17. runtime errors
18. broader argument syntax
19. slicing completion
20. object lifetime / free-list / RC改善

## P4 — optimization

21. source size
22. runtime steps
23. tape memory
24. large AtCoder corpus

---

# 18. 「完成」の定義

第一の完成条件は、Python言語の全仕様を100%再現することではありません。

以下を満たした時点を「実用的なPython→Brainfuck transpiler v1」とします。

- AtCoderで一般的なPythonコードを専用構文へ書き換えずcompileできる
- `[0] * n` 等のruntime-sized listが動く
- mutable assignmentがPythonのalias semanticsと一致する
- nested listを扱える
- function / recursionが動く
- int/string/list/tuple/dict/setの基本操作が動く
- input/printが普通のPythonコードと同じ形で使える
- generated BFのみで実行できる
- CPython differential test corpusがgreen
- comprehension等の発展的syntaxを使わなくても同値処理を基本構文で記述できる

特に、ユーザーがtranslatorの都合に合わせてPythonコードを崩す必要がある状態は完成とはみなしません。

---

# 19. 次に着手する具体タスク

次の開発PRでは、他のsyntax追加より先に以下を行います。

1. 現在のlist variable直接配置モデルを調査し、object-handle化する境界を決定
2. BF heap traversal primitiveを実装
3. dynamic list object headerを実装
4. list variableをreferenceへ変更
5. `b = a` のalias testを追加
6. `[literal] * runtime_int` をlowering
7. `[0] * n` を実BFで実行
8. nested list allocationを追加
9. AtCoder DP / graph初期化fixtureを追加
10. full CI greenを確認

このP0が通るまでは、comprehension等のsyntax sugarや便利built-inの追加を優先しません。
