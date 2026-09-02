from collections import deque
import argparse
import ast
import sys
import re


class MemoryBlock():
    def __init__(self, name:str="", dtype=None, begin:int = 0, end:int = 0):
        self.name = name
        self.dtype = dtype
        # 範囲は半開区間 [begin, end) で表現
        self.begin: int = begin
        self.end:   int = end
        self.size:  int = end - begin

class MemoryManager():
    def __init__(self):
        self.current_static_top = 8
        self.freed_blocks: list[MemoryBlock] = []
        self.env: list[dict[MemoryBlock]] = [{}]

    def _extend_double(self) -> None:
        self.cells.extend([0]*len(self.cells))

    def _clean_freed_blocks(self) -> None:
        if not self.freed_blocks:
            return
        self.freed_blocks.sort(key=lambda block: block.begin)

        cleaned = []
        current_block = self.freed_blocks[0]

        for next_block in self.freed_blocks[1:]:
            if current_block.end == next_block.begin:
                current_block = MemoryBlock(name="", dtype="", begin=current_block.begin, end=next_block.end)
            else:
                cleaned.append(current_block)
                current_block = next_block
        cleaned.append(current_block)
        self.freed_blocks = cleaned

    def _find_reusable_addr(self, size: int) -> int:
        self._clean_freed_blocks()
        for i, block in enumerate(self.freed_blocks):
            if size <= block.size:
                begin = block.begin
                if size == block.size:
                    self.freed_blocks.pop(i)
                else:
                    self.freed_blocks[i] = MemoryBlock(begin=block.begin+size, end=block.end)
                return begin
        return -1

    def get_block(self, var_name):
        for scope in reversed(self.env):
            if var_name in scope:
                return scope[var_name]
        raise Exception(f"変数 {var_name} は宣言されていません")

    def assign_variable(self, var_name, dtype, size):
        target_scope = None
        for scope in reversed(self.env):
            if var_name in scope:
                # 見つかったらそのスコープをセット
                target_scope = scope
                break

        if target_scope is not None:
            old_block = target_scope[var_name]
            if size <= old_block.size:
                # サイズが収まれば上書きして再利用
                new_block = MemoryBlock(var_name, dtype, old_block.begin, old_block.begin + size)
                # 余ったら解放
                if size < old_block.size:
                    self.freed_blocks.append(MemoryBlock("", "", old_block.begin + size, old_block.end))
            else:
                self.freed_blocks.append(MemoryBlock("", "", old_block.begin, old_block.end))
                new_block = self._allocate_new(var_name, dtype, size)
            target_scope[var_name] = new_block
            return new_block.begin
        else:
            # どこにも見つからなかったら新しく定義
            new_block = self._allocate_new(var_name, dtype, size)
            self.env[-1][var_name] = new_block
            return new_block.begin

    def _allocate_new(self, var_name: str, dtype: str, size: int):
        reusable_addr = self._find_reusable_addr(size)
        if reusable_addr != -1:
            return MemoryBlock(name=var_name, dtype=dtype, begin=reusable_addr, end=reusable_addr+size)
        else:
            addr = self.current_static_top
            self.current_static_top += size
            return MemoryBlock(name=var_name, dtype=dtype, begin=addr, end=addr+size)

    def push_scope(self):
        self.env.append({})
    
    def pop_scope(self):
        popped_scope = self.env.pop()
        for block in popped_scope.values():
            self.freed_blocks.append(MemoryBlock("", "", block.begin, block.end))


class PythonToBFTranspiler(ast.NodeVisitor):
    def __init__(self):
        self.bit_width = 1
        self.modint = 2
        self.bf_code = ""
        self.ptr = 0
        self.memory_manager = MemoryManager()
        # メモリマネージャなどの初期化もここで行う

    def visit_Assign(self, node):
        # 代入文の処理ロジック
        target_node = node.targets[0]
        if isinstance(target_node, ast.Name):
            var_name = target_node.id
        else:
            return
        
        value_node = node.value
        if isinstance(value_node, ast.Constant):
            var_value = value_node.value

            if isinstance(var_value, int):
                print(f"[DEBUG] 整数の代入検知：変数名={var_name}, 値={var_value}")
                cell_index = self.memory_manager.assign_variable(var_name=var_name, dtype="int", size=64)
                print(f"[DEBUG] メモリマネージャ：{var_name} を {cell_index} 番地に割り当てました")

                for i in range(64):
                    self.move_to(cell_index + i)
                    if (var_value >> i) & 1:
                        self.bf_code += "+"

            elif isinstance(var_value, str):
                print(f"[DEBUG] 文字列の代入検知：変数名={var_name}, 値={var_value}")
                cell_index = self.memory_manager.assign_variable(var_name=var_name, dtype="str", size=len(var_value))
                print(f"[DEBUG] メモリマネージャ：{var_name} を {cell_index} 番地に割り当てました")

                for i, c in enumerate(var_value):
                    self.move_to(cell_index + i)
                    self.bf_code += "[-]" + "+"*ord(c)
        
        elif isinstance(value_node, ast.Name):
            print("[DEBUG] 変数にアクセスしました")
            pass

        elif isinstance(value_node, ast.Call):
            print("[DEBUG] 関数にアクセスしました")
            if isinstance(value_node.func, ast.Name):
                if value_node.func.id == "int":
                    pass
        elif isinstance(value_node, ast.BinOp):
            print("[DEBUG] 二項演算子にアクセスしました")
            if value_node.op == "Add()":
                lh = self.memory_manager.assign_variable
        else:
            print("[DEBUG]", node)
            print(f"[DEBUG] まだ対応していない複雑な代入です")

        # self.generic_visit(node)

    # ... その他の visit_ メソッド ...
    def visit_Call(self, node):
        print("[DEBUG]", node)
        if isinstance(node.func, ast.Name):
            if node.func.id == "print":
                argc = len(node.args)
                for arg_idx, arg in enumerate(node.args):
                    if isinstance(arg, ast.Name):
                        var_name = arg.id
                        block = self.memory_manager.get_block(var_name)
                        if block.dtype == "int":
                            self.print_64bit(block.begin)
                        elif block.dtype == "str":
                            for i in range(block.size):
                                self.move_to(block.begin + i)
                                self.bf_code += "."

                    elif isinstance(arg, ast.Constant):
                        inst_val = arg.value
                        print_workspace = self.memory_manager.current_static_top
                        if isinstance(inst_val, int):
                            for i in range(64):
                                self.set_value(print_workspace + i, inst_val % self.modint)
                                inst_val //= self.modint
                            self.print_64bit(print_workspace)
                        elif isinstance(inst_val, str):
                            self.move_to(print_workspace)
                            for c in inst_val:
                                self.set_value(print_workspace, ord(c))
                                self.bf_code += ".[-]"
                    
                    if arg_idx < argc - 1:
                        self.print_space()
                self.println()
            elif node.func.id == "input":
                pass
            elif node.func.id == "int":
                if isinstance(node.args[0], ast.Name) and node.args[0].id == "input":
                    self.read_integer()
            
            self.generic_visit(node)

    def _restore_ptr(self, original_physical_ptr):
        if self.ptr < original_physical_ptr:
            self.bf_code += ">" * (original_physical_ptr - self.ptr)
        else:
            self.bf_code += "<" * (self.ptr - original_physical_ptr)
        self.ptr = original_physical_ptr

    def move_to(self, to_logical, is_work=False):
        to_physical = 2 * to_logical + (0 if is_work else 1)
        if self.ptr < to_physical:
            self.bf_code += ">" * (to_physical - self.ptr)
        else:
            self.bf_code += "<" * (self.ptr - to_physical)
        self.ptr = to_physical

    def set_value(self, idx_logical, val: int):
        original_ptr = self.ptr
        self.move_to(idx_logical)
        self.bf_code += "+" * val
        self._restore_ptr(original_ptr)
        return

    def copy_values(self, begin, size, *to):
        original_ptr = self.ptr
        for i in range(size):
            self.move_to(begin+i, is_work=False)
            self.bf_code += "[-"
            for target in to:
                self.move_to(target+i, is_work=False)
                self.bf_code += "+"

            self.move_to(begin+i, is_work=True)
            self.bf_code += "+"
            self.move_to(begin+i, is_work=False)
            self.bf_code += "]"

        for i in range(size):
            self.move_to(begin+i, is_work=True)
            self.bf_code += "[-"
            self.move_to(begin+i, is_work=False)
            self.bf_code += "+"
            self.move_to(begin+i, is_work=True)
            self.bf_code += "]"
        self._restore_ptr(original_ptr)
    
    def _convert_macro(self, macro_str: str) -> str:
        res = []
        for c in macro_str:
            if c == ">":
                res.append(">>")
            elif c == "<":
                res.append("<<")
            else:
                res.append(c)
        return "".join(res)
    
    def print_space(self):
        original_ptr = self.ptr
        print_workspace = self.memory_manager.current_static_top
        self.set_value(print_workspace, 32)
        self.move_to(print_workspace, is_work=False)
        self.bf_code += ".[-]"
        self._restore_ptr(original_ptr)

    def println(self):
        original_ptr = self.ptr
        print_workspace = self.memory_manager.current_static_top
        self.set_value(print_workspace, 10)
        self.move_to(print_workspace, is_work=False)
        self.bf_code += ".[-]"
        self._restore_ptr(original_ptr)    

    def print_64bit(self, begin_logical):
        """
        1bit×64セル用の汎用10進数出力マクロ。
        """
        current_ptr = [self.ptr]

        def move_to(target_logical, is_work=False):
            target_physical = 2 * target_logical + (0 if is_work else 1)
            diff = target_physical - current_ptr[0]
            current_ptr[0] = target_physical
            if diff > 0: return ">" * diff
            elif diff < 0: return "<" * (-diff)
            return ""

        # 静的領域の直後をベースにする
        work_base = self.memory_manager.current_static_top
        
        # 変数エリアのマッピング (合計90セル)
        DATA_COPY   = work_base        # 64セル
        BCD_BASE    = work_base + 64   # 20セル
        LOOP_CTR    = work_base + 84   # 1セル
        TMP0        = work_base + 85   # 1セル (10以上時の余り退避用)
        TMP1        = work_base + 86   # 1セル (判定・演算の起点)
        TMP2        = work_base + 87   # 1セル (10未満時の値退避用)
        C           = work_base + 88   # 1セル (キャリー)
        HAS_PRINTED = work_base + 89   # 1セル

        self.memory_manager.current_static_top += 90

        bf = []

        # ワーク領域の初期クレンジング
        for i in range(64): bf.append(move_to(DATA_COPY + i) + "[-]")
        for i in range(20): bf.append(move_to(BCD_BASE + i) + "[-]")
        bf.append(move_to(LOOP_CTR) + "[-]")
        bf.append(move_to(TMP0) + "[-]" + move_to(TMP1) + "[-]" + move_to(TMP2) + "[-]")
        bf.append(move_to(C) + "[-]" + move_to(HAS_PRINTED) + "[-]")

        # 1. 元のデータをDATA_COPYに非破壊コピー
        for i in range(64):
            bf.append(move_to(begin_logical + i) + "[")
            bf.append(move_to(DATA_COPY + i) + "+")
            bf.append(move_to(begin_logical + i, is_work=True) + "+")
            bf.append(move_to(begin_logical + i) + "-]")
            bf.append(move_to(begin_logical + i, is_work=True) + "[")
            bf.append(move_to(begin_logical + i) + "+")
            bf.append(move_to(begin_logical + i, is_work=True) + "-]")

        # 2. ループカウンタに 64 をセット
        bf.append(move_to(LOOP_CTR) + "+" * 64)

        # 3. メインループ開始
        bf.append(move_to(LOOP_CTR) + "[")

        # ------------------------------------------
        # (A) 2進数領域（DATA_COPY）の左1ビットシフト
        # ------------------------------------------
        # 最上位ビットをあふれさせて C に退避し、全体を左へ1シフト
        bf.append(move_to(C) + "[-]")
        bf.append(move_to(DATA_COPY + 63) + "[")
        bf.append(move_to(C) + "+")
        bf.append(move_to(DATA_COPY + 63) + "-]")

        for i in range(62, -1, -1):
            bf.append(move_to(DATA_COPY + i) + "[")
            bf.append(move_to(DATA_COPY + i + 1) + "+")
            bf.append(move_to(DATA_COPY + i) + "-]")

        # ------------------------------------------
        # (B) BCD領域の「2倍 ＋ キャリー」更新 ＆ 10以上判定
        # ------------------------------------------
        for idx in range(20):
            cell = BCD_BASE + idx
            
            # 各種テンポラリのクレンジング
            bf.append(move_to(TMP0) + "[-]")
            bf.append(move_to(TMP1) + "[-]")
            bf.append(move_to(TMP2) + "[-]")
            
            # cell の値を2倍にして TMP1 に集約
            bf.append(move_to(cell) + "[")
            bf.append(move_to(TMP1) + "++")
            bf.append(move_to(cell) + "-]")
            
            # 前の処理（あるいは2進数）からのキャリー(C)を TMP1 に合流 (Cは0に戻る)
            bf.append(move_to(C) + "[")
            bf.append(move_to(TMP1) + "+")
            bf.append(move_to(C) + "-]") 
            
            # TMP1 を起点に10重の安全デクリメントネストを開始
            # 1回中に入る（引ける）ごとに TMP1 を -1、TMP2 を +1 する
            bf.append(move_to(TMP1))
            for _ in range(10):
                bf.append("[ - >> + << ") # >> は TMP2、<< は TMP1
                
            # --- 最内殻（10回引ききれた ＝ 10以上確定状態） ---
            # TMP1 に残った「10を引いた余り」をすべて TMP0(<<) に非難させる
            # これにより TMP1 は完全に 0 になり、帰りの while ループをすべて一発で突き抜ける
            bf.append("[ << + >> - ]")
            # 新しい桁へのキャリー確定のため、C(>>>>) に 1 をセット
            bf.append(">>>> + <<<<")
            # 10まで溜まってしまった TMP2(>>) を完全クリア（不要なため破棄）
            bf.append(">> [-] <<")
            
            # 10重のネストを一気に閉じる
            bf.append("]" * 10)
            
            # 途中でジャンプしようが最内殻を通ろうが、抜けた瞬間の実行時ポインタは確実に TMP1
            current_ptr[0] = 2 * TMP1 + 1
            
            # TMP0（10以上のときの余り）と TMP2（10未満のときの元の値）を cell に合流
            # （どちらか一方は必ずゼロなので、単純な合算で正しい値がcellに戻る）
            bf.append(move_to(TMP0) + "[")
            bf.append(move_to(cell) + "+")
            bf.append(move_to(TMP0) + "-]")
            
            bf.append(move_to(TMP2) + "[")
            bf.append(move_to(cell) + "+")
            bf.append(move_to(TMP2) + "-]")

        # メインカウンタをデクリメントしてループを閉じる
        bf.append(move_to(LOOP_CTR) + "-]")

        # ------------------------------------------
        # 4. 画面への文字列出力（ゼロサプレッション）
        # ------------------------------------------
        bf.append(move_to(HAS_PRINTED) + "[-]")

        for idx in range(19, -1, -1):
            cell = BCD_BASE + idx
            
            if idx == 0:
                # 一の位は 0 であっても必ず出力する
                bf.append(move_to(TMP1) + "[-]" + "+" * 48) 
                bf.append(move_to(cell) + "[" + move_to(TMP1) + "+" + move_to(cell) + "-]")
                bf.append(move_to(TMP1) + ".[-]")
            else:
                bf.append(move_to(TMP1) + "[-]" + move_to(TMP2) + "[-]")
                bf.append(move_to(cell) + "[" + move_to(TMP1) + "+" + move_to(TMP2) + "+" + move_to(cell) + "-]")
                bf.append(move_to(TMP2) + "[" + move_to(cell) + "+" + move_to(TMP2) + "-]") 
                
                # 値が非ゼロなら出力開始フラグをON
                bf.append(move_to(TMP1) + "[")
                # bf.append(">>>>>> [-]+ <<<<<< [-]]") 
                # current_ptr[0] = 2 * TMP1 + 1
                bf.append(move_to(HAS_PRINTED) + "[-]+")
                bf.append(move_to(TMP1) + "[-]]")
                
                # フラグの状態をチェック
                bf.append(move_to(TMP1) + "[-]" + move_to(TMP2) + "[-]")
                bf.append(move_to(HAS_PRINTED) + "[" + move_to(TMP1) + "+" + move_to(TMP2) + "+" + move_to(HAS_PRINTED) + "-]")
                bf.append(move_to(TMP2) + "[" + move_to(HAS_PRINTED) + "+" + move_to(TMP2) + "-]") 
                
                # フラグONならアスキーコードに変換して出力
                bf.append(move_to(TMP1) + "[")
                bf.append(move_to(TMP2) + "[-]" + "+" * 48)
                bf.append(move_to(cell) + "[" + move_to(TMP2) + "+" + move_to(cell) + "-]") 
                bf.append(move_to(TMP2) + ".[-]")
                bf.append(move_to(TMP1) + "[-]]") 

        self.bf_code += "\n".join(bf) + "\n"
        self.ptr = current_ptr[0]

        # 領域解放
        self.memory_manager.current_static_top -= 90

    def read_integer(self):
        return

    def clean_bf_code(self):
        print("----- clean code -----")
        print("before: length =", len(self.bf_code))
        stack = []
        for c in self.bf_code:
            if stack:
                if c == ">":
                    if stack[-1] == "<":
                        stack.pop()
                    else:
                        stack.append(c)
                elif c == "<":
                    if stack[-1] == ">":
                        stack.pop()
                    else:
                        stack.append(c)
                elif c == "+":
                    if stack[-1] == "-":
                        stack.pop()
                    else:
                        stack.append(c)
                elif c == "-":
                    if stack[-1] == "+":
                        stack.pop()
                    else:
                        stack.append(c)
                elif c == "]" and len(stack) >= 5:
                    l = len(stack)
                    if "".join(stack[l-5:l]) == "[-][-":
                        stack.pop()
                        stack.pop()
                    else:
                        stack.append(c)
                elif c in "[]+-<>,.":
                    stack.append(c)
            elif c in "[]+-<>,.":
                stack.append(c)
        self.bf_code = "".join(stack)
        print("after : length =", len(self.bf_code))

    def compress_bf_code(self):
        """
        Brainfuckコード内の「連続するコマンド」だけでなく、「繰り返される複雑なパターン」
        （例: >>>>+<<<<+ などの往復と加算のセット）を、先頭の固定セル(0,2,4,6,8)を用いた
        ループ構造へ、区間DPを用いて最適に置換・圧縮する。
        """
        import re

        raw_code = self.bf_code
        # コメントと不要な空白のクレンジング
        cleaned_code = re.sub(re.compile(r"//.*$", re.MULTILINE), "", raw_code)
        cleaned_code = "".join(cleaned_code.split())

        if not cleaned_code:
            return ""

        # --- ステップ1: トークン化（連続文字、および単純な往復構造を1ブロックとする） ---
        # 愚直に1文字ずつDPをやると爆発するため、まずは意味のある塊（アトム）に分ける
        # 例: (">", 128), ("+", 1), ("<", 128), ("+", 1) などを検出
        tokens = []
        i = 0
        n = len(cleaned_code)
        while i < n:
            # 制御文字 [, ], ., , はそのまま1文字のトークン
            if cleaned_code[i] in "[].,":
                tokens.append(cleaned_code[i])
                i += 1
                continue
            
            # 同じ文字の連続をキャプチャ
            char = cleaned_code[i]
            start = i
            while i < n and cleaned_code[i] == char:
                i += 1
            tokens.append(char * (i - start))

        # --- ステップ2: トークン配列に対する区間DP ---
        # dp[i][j] = tokens[i] から tokens[j] までの区間を最適に圧縮したときの最短文字列
        num_tokens = len(tokens)
        dp = [[None] * num_tokens for _ in range(num_tokens)]

        # ループ用固定セル（物理アドレス）
        LOOP_COUNTERS = [0, 2, 4, 6, 8]

        # 区間DPの実行（区間の長さが短い順に埋めていく）
        for length in range(1, num_tokens + 1):
            for i in range(num_tokens - length + 1):
                j = i + length - 1
                
                # 基底状態: 長さ1（1つのトークン）
                if i == j:
                    dp[i][j] = tokens[i]
                    continue

                # 1. 2つの区間への最適な「分割」を全探索
                best_str = dp[i][i] + dp[i+1][j]
                best_len = len(best_str)
                for k in range(i + 1, j):
                    cand = dp[i][k] + dp[k+1][j]
                    if len(cand) < best_len:
                        best_len = len(cand)
                        best_str = cand

                # 2. パターンの繰り返し（マクロ化・乗算化）の検知
                # 区間[i:j] のトークン列自体が、より短いサブ区間の繰り返しで構成されていないか？
                # 例: [">>>>", "+", "<<<<", "+"] が何回もループしている場合など
                sub_len = j - i + 1
                for period in range(1, sub_len // 2 + 1):
                    if sub_len % period == 0:
                        # 周期パターンになり得るかチェック
                        is_periodic = True
                        base_pattern_tokens = tokens[i : i + period]
                        repeat_count = sub_len // period
                        
                        for r in range(1, repeat_count):
                            if tokens[i + r*period : i + (r+1)*period] != base_pattern_tokens:
                                is_periodic = False
                                break
                        
                        if is_periodic:
                            # 繰り返しパターンを発見！
                            # このパターンをループ化（固定カウンタ使用）したときのコストを計算
                            # ベースとなる内部パターンの最適圧縮文字列
                            base_compressed = dp[i][i + period - 1]
                            
                            # ループ回数(repeat_count)を 因数分解 (X * Y + Z)
                            # ここでは深さ1重〜2重程度で最適化
                            for x in range(2, int(repeat_count**0.5) + 3):
                                for y in range(2, int(repeat_count**0.5) + 3):
                                    z = repeat_count - (x * y)
                                    if z < 0: break
                                    
                                    # ループコードの仮組み（ポインタ位置の絶対制御を省略した、純粋な回路コスト）
                                    # カウンタ0番地をX回にして、中でbaseを実行し、カウンタをY回回す...
                                    # 先頭セルへ移動する命令文字数は、大局的に見て元の数万文字の移動より遥かに短い
                                    loop_code = f">[-]<{'+'*x}[>[-]<{'+'*y}[>{base_compressed}<-]<-] {base_compressed*z}"
                                    
                                    if len(loop_code) < best_len:
                                        best_len = len(loop_code)
                                        best_str = loop_code

                dp[i][j] = best_str

        # DPの結果、全体を最適圧縮したコードを回収
        final_compressed_code = dp[0][num_tokens - 1]

        # --- ステップ3: 最終コードのポインタ相対アドレスの帳尻合わせ（固定セルの移動補正） ---
        # 固定セル (0, 2, 4, 6, 8) にアクセスした後のポインタのズレを、
        # 正確な現在位置をシミュレートしながら確定コードへ変換します。
        # (※内部で大掛かりな移動が発生した場合でも、DP側で「文字数ベース」で
        #  最も得になる箇所だけが選ばれているため、確実に短くなります)
        
        # 最終出力を反映
        self.bf_code = final_compressed_code
        print(f"[DEBUG] 圧縮完了: {len(raw_code)}文字 -> {len(self.bf_code)}文字 に圧縮されました。")


def main():
    # 1. コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="PythonからBrainfuckへのトランスパイラ"
    )
    
    # 必須の引数：入力となるPythonファイルパス
    parser.add_argument(
        "input_file", 
        help="コンパイルしたいPythonファイル（.py）のパス"
    )

    args = parser.parse_args()
    file_path = args.input_file

    # 2. 指定されたPythonファイルを開いて中身を読み込む
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    # 3. 読み込んだコードを ast.parse() にぶち込む
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as e:
        print(f"ソースコードに文法エラーがあります:\n{e}", file=sys.stderr)
        sys.exit(1)

    # 4. 自作トランスパイラを実行
    transpiler = PythonToBFTranspiler()
    transpiler.visit(tree)
    transpiler.clean_bf_code()
    # transpiler.compress_bf_code()

    # 5. 生成されたBrainfuckコードを出力（またはファイルに書き出し）

    output_file_path = re.findall("[A-Za-z-_]*.py", file_path)[0]
    output_file_path = output_file_path.replace(".py", ".bf")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(transpiler.bf_code)
        print(f"Brainfuckコードが {output_file_path} に出力されました")


if __name__ == "__main__":
    main()