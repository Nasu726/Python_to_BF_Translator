# Python → Brainfuck Translator 実装計画

## 0. 最終目標

このプロジェクトの目標は、**Pythonの基本的な文法・型・制御構造・関数・可変オブジェクトの意味論を、生成されたBrainfuckだけで実行できるトランスパイラ**にすることです。

Pythonはコンパイル時にのみ使用し、生成後の `.bf` は標準Brainfuckインタプリタだけで実行できる状態を維持します。

主用途はAtCoder等の競技プログラミングです。典型的なPythonコードをtranslator向けに崩さず、そのまま入力できることを優先します。

最低限、次のようなコードは自然に動く必要があります。

```python
n = int(input())
a = [0] * n
b = a

for i in range(n):
    a[i] = i

a.sort()
print(b)
```

---

# 1. 言語仕様の基本方針

## 1.1 実装対象

「別の便利構文に頼らず普通のPythonプログラムを書くために必要な基本機能」を実装対象とします。

- 代入・参照・スコープ
- 基本リテラル
- `None`
- int / bool / str
- list / tuple / dict / set
- 算術・比較・論理・bit演算
- 添字アクセス・更新
- `if` / `while` / `for`
- `break` / `continue`
- 関数定義・呼び出し・`return`
- 再帰
- mutable objectの参照意味論
- 行単位入力・出力
- Pythonコードで通常必要になる基本操作

## 1.2 初期段階では実装しない発展的構文

基本構文で同値処理を書けるsyntax sugarは優先しません。

- list / dict / set comprehension
- generator expression
- lambda
- decorator
- generator / `yield`
- async / await
- pattern matching (`match`)
- walrus operator (`:=`)
- metaclass等の高度なobject model

この非対応は「基本的な処理能力を欠く」ことを意味してはいけません。たとえばlist comprehensionを実装しなくても、通常の`for`と`append`で同じ処理が書ける必要があります。

## 1.3 標準ライブラリ

標準ライブラリ全体の再実装は目標にしません。

`collections`, `heapq`, `bisect`, `itertools`, `math` 等は別レイヤの課題です。

一方で、言語を普通に使うために不可欠なものはruntime primitiveまたはcompiler intrinsicとして提供します。

優先例:

- `input`
- `print`
- `int`
- `str`
- `bool`
- `len`
- `range`
- `deepcopy`（後述のcopy semantics用intrinsic）

`sum`, `enumerate`, `zip` 等は基本構文で同値処理を書けるため後回しにできます。ただし`list.sort()`のように、代替実装をユーザーへ毎回書かせると実用性を大きく損なう基本メソッドは優先します。

---

# 2. 明示的な意味論方針

## 2.1 list代入はPython互換のalias semanticsを維持する

次はcopyではありません。

```python
a = [1, 2]
b = a
b[0] = 9
print(a)
# [9, 2]
```

`b = a` はPython同様、同一list objectへの参照共有とします。

これは「shallow copy」とも異なります。単なるaliasです。

## 2.2 shallow copy

以下をshallow copyとして実装します。

```python
b = a.copy()
b = a[:]
```

外側のlist objectだけを複製し、要素がobject referenceなら参照は共有します。

例:

```python
a = [[1], [2]]
b = a.copy()
b[0].append(3)
print(a)
# [[1, 3], [2]]
```

## 2.3 deep copy

競プロではnested listの参照共有が事故原因になりやすいため、標準ライブラリ全体を実装する前でも**`copy.deepcopy`相当のintrinsic**を提供します。

基本APIは次を予定します。

```python
b = deepcopy(a)
```

将来的に`import copy`対応レイヤを作る場合は、`copy.deepcopy(a)`を同じintrinsicへloweringしてもよいものとします。

### deepcopyの意味論

最終的には単なる再帰コピーではなく、Pythonの`copy.deepcopy`に近い**memo付きobject graph copy**を目標にします。

- scalar immutable値はそのまま
- list / dict / set / tuple等を再帰複製
- 同じsource objectが複数箇所から参照されている場合、clone側でも同じclone objectを共有
- cycleを含むobject graphでも無限再帰しない

例:

```python
x = [1]
a = [x, x]
b = deepcopy(a)

b[0].append(2)
print(b[1])
# [1, 2]

print(a)
# [[1], [1]]
```

P0ではまずlist/tupleを中心としたacyclic graphで動かし、その後memo tableを追加してshared substructure/cycleまで完成させます。

## 2.4 list repetitionのPython意味論

```python
a = [[0] * m] * n
```

はPython同様、**同じinner listへの参照をn個並べる**ものとします。

独立行が必要なら、comprehension非対応期間は基本構文で次のように書けます。

```python
a = []
for i in range(n):
    a.append([0] * m)
```

あるいは既存objectを完全複製したい場合は`deepcopy`を使用できます。

---

# 3. 現在の基盤

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
n = int(input())
a = [0] * n
b = a
```

この2行を正しく扱うには、list演算を個別追加するだけでは不十分で、**runtime object / reference / heap model**が必要です。

---

# 4. Phase 1 — Runtime object / heap / reference基盤

## 目的

mutable objectを「変数領域そのもの」ではなくruntime objectとして管理します。

## 必須要件

- scalar valueとobject referenceを区別
- list等のmutable valueはheap objectとして保持
- 変数はobject handle/referenceを保持
- `b = a` は同一objectを指す
- object identityを保持
- runtime length / capacityを持つ
- compile-timeに長さが分からなくても生成可能
- Brainfuck自身がheapを操作
- Python runtimeへ依存しない

## 想定heap object header

概念的には以下を持たせます。

```text
[type]
[length]
[capacity / block size]
[allocation metadata]
[data ...]
```

BFでは通常のmachine pointerを直接使えないため、handleからobjectへ到達するためのmarker traversal / indexed traversal primitiveをruntimeとして定義します。

## allocator

実装順:

1. monotonic / bump allocation
2. object handle
3. reusable free-list
4. ownership/lifetime tracking
5. 必要に応じreference counting
6. cycle GCは後回し

競プロコードではGCの完全性より、loop内で一時objectを生成してもmemoryを一方的に消費し続けないことを優先します。

## Acceptance tests

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

最後のケースで単純heap leakを続けないこと。

---

# 5. Phase 2 — Dynamic list

Heap基盤の直後に実装します。

## 必須構文

```python
a = []
a = [1, 2, 3]
a = [0] * n
a = [x] * n
```

runtime値によるrepeatは必須です。

```python
n = int(input())
a = [0] * n
```

## 必須操作

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
- concatenation `a + b`
- repetition `a * n`, `n * a`
- alias assignment
- `a.copy()` shallow copy
- `a[:]` shallow copy
- `deepcopy(a)`

## index / slice

```python
a[i]
a[-1]
a[i] = x
```

runtime indexへ対応します。

basic indexing安定後:

```python
a[l:r]
a[l:r:s]
a[::-1]
```

slice assignmentは後段で構いません。

## nested list

AtCoderでは2次元配列・隣接リストが必須です。

```python
a = []
for i in range(n):
    a.append([0] * m)
```

```python
g = []
for i in range(n):
    g.append([])

g[u].append(v)
```

を必須acceptance caseとします。

---

# 6. Phase 2.5 — list.sort / stable sorting runtime

`list.sort()` は競プロ実用性に直結するため、dict/setより大幅に優先します。

## 採用アルゴリズム

**bottom-up merge sort** を標準runtime実装とします。

理由:

- stable
- worst-case `O(n log n)`
- pivot悪化が無い
- recursion不要でBFにloweringしやすい
- 比較器を差し替えやすい
- predictableなcontrol flow

quicksortは平均`O(n log n)`でも最悪`O(n^2)`で、通常はstableでもないため標準実装には採用しません。

## 基本contract

```python
a.sort()
```

- in-place
- stable
- worst-case `O(n log n)` comparator calls
- auxiliary buffer `O(n)`
- Python同様、戻り値は`None`

そのため`None`を基本value modelへ追加します。

## 最初に対応するelement type

1. `list[int]`
2. `list[str]`
3. tupleのlexicographic comparison
4. nested comparable container

型推論でelement comparatorをcompile-time選択し、汎用tag dispatchを毎比較で行わない構造を優先します。

## reverse

早期に次も対応します。

```python
a.sort(reverse=True)
```

`reverse`がruntime boolの場合も最終的には対応します。

## key

AtCoderでは次も重要です。

```python
a.sort(key=f)
```

lambdaは発展構文として非目標ですが、named functionを`key=`へ渡せるようにします。

function runtime完成後に実装し、Python同様、原則としてkeyは各要素について一度だけ評価してcacheします。

概念的には:

```text
[(key0, ref0), (key1, ref1), ...]
```

をtemporary bufferへ作りstable merge sortします。

これによりkeyが重い場合でも`O(n log n)`回keyを再計算しません。

## sorted

`sorted(a)` は`list.sort()` runtimeを再利用して実装できますが、built-inなので後段でも構いません。

実装する場合:

```text
shallow-copy iterable → list.sort-compatible runtime → new listを返す
```

とします。

## BF runtime design

sort対象listとは別に、同じ要素数を保持できるtemporary merge bufferをheapへ確保します。

bottom-up run width:

```text
1, 2, 4, 8, ...
```

で隣接runをmergeし、各pass後にsource/destination bufferをswapします。

最後のsorted dataがtemporary側にある場合のみ元listへ戻します。

## Acceptance tests

```python
a = [5, 1, 4, 1, 3]
a.sort()
print(a)
# [1, 1, 3, 4, 5]
```

```python
a = [5, 1, 4, 1, 3]
a.sort(reverse=True)
print(a)
# [5, 4, 3, 1, 1]
```

stable性はkey付きrecordで検証します。

```python
a = [(2, 0), (1, 1), (2, 2), (1, 3)]

def first(x):
    return x[0]

a.sort(key=first)
print(a)
# [(1, 1), (1, 3), (2, 0), (2, 2)]
```

---

# 7. Phase 3 — Compiler-side type / value model

Brainfuckにはruntime型システムがないため、コンパイル時に可能な限り型を確定します。

## 基本方針

- scalarは固定width
- mutable objectはreference
- `None`はsingleton value
- container element typeはcompile-time inference優先
- 普通の競プロコードで自然に推論できることを目標にする
- 最初から全値をtagged unionにしてBF runtimeを重くしない

対象:

- None
- int
- bool
- str
- list[T]
- tuple[T...]
- dict[K,V]
- set[T]
- function signatures

必須検査:

- incompatible assignment
- function argument type propagation
- return type propagation
- container element type propagation
- nested container type

---

# 8. Phase 4 — Tuple / unpacking / assignment

必須:

```python
a, b = b, a
x, y = pair
```

- tuple literal
- tuple value
- unpack
- nested unpack
- multiple return

```python
def f():
    return 1, 2

a, b = f()
```

starred unpackは後回し。

---

# 9. Phase 5 — String

現在のfixed byte stringを通常のPythonコードで必要な操作まで拡張します。

必須:

- assignment
- equality / ordering
- len
- index / negative index
- slice
- concatenation
- repetition
- iteration
- membership
- `str(int)`
- `int(str)`

Brainfuck I/Oがbyte単位なので、まずUTF-8 byte sequenceとして扱い、AtCoderで主に使うASCII入力についてPythonコードと同じ結果を保証します。

完全なUnicode code point semanticsは別フェーズ。

---

# 10. Phase 6 — Function / Scope / Call stack

AtCoderで普通のPythonコードを使うための最重要項目です。

```python
def f(x, y):
    z = x + y
    return z

ans = f(a, b)
```

必須runtime:

- call frame
- locals
- parameters
- return value
- nested calls
- recursion

recursion acceptance:

```python
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)
```

scope:

- local
- module/global
- `global` は必要時追加
- `nonlocal`, closureは後回し

引数:

1. positional
2. default
3. keyword
4. `*args`, `**kwargs` は低優先

このphase完了後、`list.sort(key=named_function)`を実装します。

---

# 11. Phase 7 — Control flow完成

以下を普通に扱います。

```python
for i in range(n):
for i in range(l, r):
for i in range(l, r, step):
for x in a:
for ch in s:
```

`range`のstart/stop/stepはruntime値対応にします。

- nested break/continue
- loop else
- return through nested control flow

を統一loweringで処理します。

---

# 12. Phase 8 — Operators完成

## int

- `+ - * // % **`
- unary `+ -`
- `& | ^ ~ << >>`
- comparison

## comparison

- chained comparison
- `in`
- `not in`
- lexicographic tuple/list/string comparison

## logical

- `and`
- `or`
- `not`
- short-circuit
- operand-value return semantics

## identity

object handle導入後に`is` / `is not`を実装します。

---

# 13. Phase 9 — Dict / Set

AtCoderで頻出するため基本containerとして実装します。

## dict

```python
d = {}
d[key] = value
x = d[key]
key in d
len(d)
for key in d:
```

correctness-onlyのlinear tableから始めてもよいですが、実用版はhash tableへ進めます。

候補:

- open addressing
- tombstone
- resize
- fixed-width hash

## set

- add
- remove / discard
- membership
- len
- iteration

同じhash runtimeを共有します。

---

# 14. Phase 10 — Numeric types

## int

当面signed 64-bitを正式ABIとして維持します。

Python arbitrary precision intは非常に高コストなため、AtCoder用途ではint64を仕様差として認めます。

## float

後段でfloat64を追加します。

- literal
- conversion
- `+ - * /`
- comparison
- print / parse

Python意味論一致を優先するならIEEE-754 binary64 software runtimeを採用します。

---

# 15. Phase 11 — Exceptions / Error model

valid contest inputでは優先度は低めですが、silent zero-fill等を最終仕様にはしません。

最低限:

- division by zero
- index out of range
- invalid unpack
- invalid int conversion
- missing dict key

にruntime error stateを持たせます。

完全な`try/except`は後回しでも、誤値のまま処理継続しないことを保証します。

---

# 16. Phase 12 — Memory management完成

Heap/object system安定後:

1. compile-time lifetime analysis
2. stack/region allocation可能objectはstackへ
3. escaping mutable objectだけheapへ
4. heap free-list
5. reference counting
6. cycle必要時のみGC検討

`deepcopy`のmemo tableやsort temporary bufferもこのallocator上で安全にallocate/freeできる必要があります。

---

# 17. Optimization

Correctnessの後に実施します。

## source size

- repeated primitive sharing
- runtime loop化
- pointer movement削減
- base-4 / bit-pair backend
- constant propagation
- dead temporary elimination

## runtime

- arithmetic primitive改善
- traversal回数削減
- object layout locality
- list index traversal改善
- merge sort buffer locality
- comparator specialization
- decimal I/O改善

## memory

- liveness reuse
- frame reuse
- heap block reuse
- sort temporary buffer reuse

測定値:

- generated BF command count
- executed BF primitive steps
- peak tape cells

---

# 18. Differential testing

主要機能は同一source/inputを

1. CPython
2. Python → BF → BF interpreter

で実行し、stdoutを比較します。

意図的な仕様差を除き、不一致はregressionです。

## AtCoder compatibility corpus

最低限以下をfixture化します。

### input

```python
n = int(input())
a, b = map(int, input().split())
a = list(map(int, input().split()))
s = input()
```

### DP

```python
dp = [0] * (n + 1)
```

### alias

```python
a = [0] * n
b = a
b[0] = 1
assert a[0] == 1
```

### shallow copy

```python
a = [[0], [1]]
b = a.copy()
b[0].append(2)
assert a[0] == [0, 2]
```

### deep copy

```python
a = [[0], [1]]
b = deepcopy(a)
b[0].append(2)
assert a[0] == [0]
```

### 2D array

```python
a = []
for i in range(h):
    a.append([0] * w)
```

### graph

```python
g = []
for i in range(n):
    g.append([])

g[u].append(v)
```

### sort

```python
a = list(map(int, input().split()))
a.sort()
print(a)
```

### stable sort

```python
a = [(2, 0), (1, 1), (2, 2), (1, 3)]

def first(x):
    return x[0]

a.sort(key=first)
```

### function

```python
def dfs(v):
    ...
```

各milestoneで「Python側の書き換え無し」にcompile/runできることをacceptance conditionとします。

---

# 19. 実装優先順位

## P0 — AtCoder実用性を決める基盤

1. runtime object handle
2. heap allocator
3. mutable alias semantics
4. dynamic list length/capacity
5. `[x] * n`
6. nested list
7. `a.copy()` / `a[:]` shallow copy
8. `deepcopy(a)` basic container copy
9. stable `list.sort()` for `list[int]`
10. `list.sort(reverse=...)`

この段階で必ず次を通します。

```python
n = int(input())
a = [0] * n
b = a
for i in range(n):
    a[i] = i
a.sort(reverse=True)
print(b)
```

## P1 — 普通のアルゴリズム記述

11. function / return / call frame
12. recursion
13. runtime `range` step
14. tuple values/unpack
15. string index/slice/concat/repeat
16. membership operators
17. lexicographic comparator
18. `list.sort(key=named_function)`
19. deepcopy memoization / shared-reference preservation

## P2 — 競プロcontainer

20. nested container type inference
21. dict
22. set
23. sort comparator coverage for supported comparable types

## P3 — completeness

24. float64
25. runtime errors
26. broader argument syntax
27. slicing completion
28. object lifetime/free-list/RC改善
29. cycle-safe deepcopy

## P4 — optimization

30. source size
31. runtime steps
32. tape memory
33. merge sort / comparator optimization
34. large AtCoder corpus

---

# 20. 「完成」の定義

実用的なPython→Brainfuck transpiler v1は以下を満たすものとします。

- AtCoderで一般的なPythonコードを専用構文へ書き換えずcompileできる
- `[0] * n`等のruntime-sized listが動く
- `b = a`がPython同様aliasになる
- shallow copyとdeep copyを明確に使い分けられる
- nested listが動く
- `list.sort()`がstableかつworst-case `O(n log n)`で動く
- function / recursionが動く
- int/string/list/tuple/dict/setの基本操作が動く
- input/printが通常のPythonコードと同じ形で使える
- generated BFのみで実行できる
- CPython differential corpusがgreen
- comprehension等の発展syntax無しでも同値処理を基本構文で記述できる

ユーザーがtranslatorの都合に合わせてPythonコードを崩す必要がある状態は完成とはみなしません。

---

# 21. 次に着手する具体タスク

次の開発PRでは他のsyntax追加より先に以下を行います。

1. 現在のlist直接配置モデルからobject-handle modelへの移行境界を決定
2. BF heap traversal primitive実装
3. dynamic list object header実装
4. list variableをreferenceへ変更
5. `b = a` alias test
6. `[literal] * runtime_int` lowering
7. `[0] * n` 実BF test
8. nested list allocation
9. shallow-copy primitive
10. deepcopy primitiveの最小版
11. merge-sort temporary buffer設計
12. `list[int].sort()` bottom-up merge sort
13. reverse sort
14. AtCoder DP / graph / sort fixture
15. full CI green

P0が通るまでは、comprehension等のsyntax sugarや便利built-inの追加を優先しません。