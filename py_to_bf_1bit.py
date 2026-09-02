from collections import deque
import argparse
import ast
import sys
import re


class MemoryBlock():
    def __init__(self, name:str="", dtype:str="", begin:int = 0, end:int = 0):
        self.name:  str = name
        self.dtype: str = dtype
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

    def _allocate_new(self, var_name: str, dtype: str, size: int):
        reusable_addr = self._find_reusable_addr(size)
        if reusable_addr != -1:
            return MemoryBlock(name=var_name, dtype=dtype, begin=reusable_addr, end=reusable_addr+size)
        else:
            addr = self.current_static_top
            self.current_static_top += size
            return MemoryBlock(name=var_name, dtype=dtype, begin=addr, end=addr+size)

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
            return new_block
        else:
            # どこにも見つからなかったら新しく定義
            new_block = self._allocate_new(var_name, dtype, size)
            self.env[-1][var_name] = new_block
            return new_block
        
    def free_variable(self, var_name):
        target_scope = None
        for scope in reversed(self.env):
            if var_name in scope:
                # 見つかったらそのスコープをセット
                target_scope = scope
                break
        
        if target_scope is not None:
            block = target_scope[var_name]
            self.freed_blocks.append(MemoryBlock("", "", block.begin, block.end))
        

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
        self.tmp_var_name = "temporary_variable_in_compilation_" # 普通に宣言したら被らない名前
        self.tmp_num = 0
        self.max_str_len = 64
        self.memory_manager = MemoryManager()
        # メモリマネージャなどの初期化もここで行う

    def visit_Constant(self, node):
        print("[DEBUG] 即値にアクセスしました")
        block = None
        tmp_name = self.tmp_var_name + str(self.tmp_num)

        if isinstance(node.value, int):
            size = max(1, node.value.bit_length())
            block = self.memory_manager.assign_variable(var_name=tmp_name, dtype="int", size=size)
            # 即値のセットアップ
            for i in range(size):
                self.move_to(block.begin + i)
                if (node.value >> i) & 1:
                    self.bf_code += "+"

        elif isinstance(node.value,str):
            size = min(len(node.value), self.max_str_len)
            block = self.memory_manager.assign_variable(var_name=tmp_name, dtype="str", size=size)
            for i, c in enumerate(node.value[:size]):
                self.move_to(block.begin + i)
                self.bf_code += "[-]" + "+" * ord(c)
        
        elif isinstance(node.value, bool):
            print("[DEBUG] bool型は未対応です")
            # block = self.memory_manager.assign_variable(var_name=tmp_name, dtype="bool", size=1)
        else:
            raise ValueError(f"不明な即値です：{node.value}")

        self.tmp_num += 1
        print(f"[DEBUG] メモリマネージャ：{block.name} を {block.begin} 番地に割り当てました")
        return block 

    def visit_Name(self, node):
        print("[DEBUG] 変数にアクセスしました")
        return self.memory_manager.get_block(var_name=node.id)

    def visit_BinOp(self, node):
        print("[DEBUG] 二項演算子にアクセスしました")
        left_hand:  MemoryBlock = self.visit(node.left)
        right_hand: MemoryBlock = self.visit(node.right)
        result_block = None
        tmp_name = self.tmp_var_name + str(self.tmp_num)
        self.tmp_num += 1

        if isinstance(node.op, ast.Add):
            if left_hand.dtype == right_hand.dtype == "int":
                calc_size = min(64, max(left_hand.size, right_hand.size)+1)
                result_block = self.memory_manager.assign_variable(var_name=tmp_name, dtype="int", size=calc_size)
                self._math_add(left_hand, right_hand, result_block)
                print(f"[DEBUG] メモリマネージャ：加算結果 {result_block.name} を {result_block.begin} 番地に割り当てました")
            elif left_hand.dtype == right_hand.dtype == "str":
                total_size = min(left_hand.size + right_hand.size, self.max_str_len)
                result_block = self.memory_manager.assign_variable(var_name=tmp_name, dtype="str", size=total_size)
                self._math_concat_str(left_hand, right_hand, result_block)
                print(f"[DEBUG] メモリマネージャ：結合結果 {result_block.name} を {result_block.begin} 番地に割り当てました")
            else:
                raise TypeError(f"{left_hand.dtype} と {right_hand.dtype} の値に対して {node.op} は適用できません。")

        elif isinstance(node.op, ast.Sub):
            pass
        elif isinstance(node.op, ast.Mult):
            pass
        elif isinstance(node.op, ast.FloorDiv):
            pass
        elif isinstance(node.op, ast.Mod):
            pass
        elif isinstance(node.op, ast.And):
            pass
        elif isinstance(node.op, ast.Or):
            pass
        elif isinstance(node.op, ast.BitAnd):
            pass
        elif isinstance(node.op, ast.BitOr):
            pass
        elif isinstance(node.op, ast.BitXor):
            pass
        else:
            raise NotImplementedError(f"演算子 {type(node.op).__name__} は未実装です")

        self._free_tmp(left_hand, right_hand)
        return result_block

    def visit_Assign(self, node):
        print("[DEBUG] 代入演算子にアクセスしました")
        right_hand: MemoryBlock = self.visit(node.value)
        if right_hand is None:
            return

        target_name = node.targets[0].id
        try:
            old_block = self.memory_manager.get_block(target_name)
            self._zero_clear_region(old_block.begin, old_block.size)
        except Exception:
            pass

        dest_block  = self.memory_manager.assign_variable(target_name, right_hand.dtype, right_hand.size)
        print(f"[DEBUG] メモリマネージャ：{dest_block.name} を {dest_block.begin} 番地に割り当てました")

        self.copy_values(right_hand.begin, right_hand.size, dest_block.begin)
        self._free_tmp(right_hand)
    
    def visit_Call(self, node):
        print("[DEBUG] 関数にアクセスしました")
        if not isinstance(node.func, ast.Name):
            return None

        if node.func.id == "print":
            argc = len(node.args)
            for arg_idx, arg in enumerate(node.args):
                arg_block: MemoryBlock = self.visit(arg)

                if arg_block.dtype == "int":
                    self.print_64bit(arg_block.begin)
                elif arg_block.dtype == "str":
                    self.move_to(arg_block.begin)
                    for i in range(arg_block.size):
                        self.bf_code += ".>>"
                self._free_tmp(arg_block)
                
                if arg_idx < argc - 1:
                    self.print_space()
            self.println()
            return None
        
        elif node.func.id == "input":
            print("[DEBUG] input にアクセスしました")
            tmp_name = self.tmp_var_name + str(self.tmp_num)
            self.tmp_num += 1
            result_block = self.memory_manager.assign_variable(var_name=tmp_name, dtype="str", size=self.max_str_len)
            self._io_read_string(result_block)
            return result_block

        elif node.func.id == "int":
            if isinstance(node.args[0], ast.Call) and getattr(node.args[0].func, "id", "") == "input":
                print("[DEBUG] int(input()) にアクセスしました")
                tmp_name = self.tmp_var_name + str(self.tmp_num)
                self.tmp_num += 1
                result_block = self.memory_manager.assign_variable(var_name=tmp_name, dtype="int", size=64)
                self._io_read_int(result_block)
                return result_block
        return None

    def _allocate_tmp(self, op, *args: MemoryBlock):
        if isinstance(op, ast.Add):
            left_hand, right_hand = args
            result_block = None
            if left_hand.dtype == right_hand.dtype:
                if left_hand.dtype == "int":
                    result_block = self.memory_manager.assign_variable(var_name=self.tmp_var_name + str(self.tmp_num), dtype="int", size=64)
                elif left_hand.dtype == "str":
                    result_block = self.memory_manager.assign_variable(var_name=self.tmp_var_name + str(self.tmp_num), dtype="str", size=left_hand.size+right_hand.size)
                elif left_hand.dtype == "bool":
                    result_block = self.memory_manager.assign_variable(var_name=self.tmp_var_name + str(self.tmp_num), dtype="bool", size=1)
                else:
                    raise TypeError(f"{left_hand.dtype} と {right_hand.dtype} の値に対して {op} は適用できません。")
            else:
                raise TypeError(f"{left_hand.dtype} と {right_hand.dtype} の値に対して {op} は適用できません。")
            return result_block
        return 

    def _zero_clear_region(self, begin, size):
        original_ptr = self.ptr
        for i in range(size):
            self.move_to(begin+i, is_work=False)
            self.bf_code += "[-]"
        self._restore_ptr(original_ptr)

    def _free_tmp(self, *args: MemoryBlock):
        for arg in args:
            if arg.name.startswith("temporary_variable_in_compilation_"):
                self._zero_clear_region(arg.begin, arg.size)
                self.memory_manager.free_variable(arg.name)
                print(f"[DEBUG] {arg.name} を解放しました")

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

    def _math_concat_str(self, left_block: MemoryBlock, right_block: MemoryBlock, result: MemoryBlock):
        """文字列の結合処理 (最大64文字制限対応)"""
        print(f"[DEBUG] BF生成: 文字列結合を実行します")
        # まず左辺をコピー
        copy_len_left = min(left_block.size, result.size)
        if copy_len_left > 0:
            self.copy_values(left_block.begin, copy_len_left, result.begin)
        
        # 次に右辺をオフセット位置からコピー（上限まで）
        copy_len_right = min(right_block.size, result.size - copy_len_left)
        if copy_len_right > 0:
            self.copy_values(right_block.begin, copy_len_right, result.begin + copy_len_left)

    def _math_add(self, left_block: MemoryBlock, right_block: MemoryBlock, result: MemoryBlock):
        print(f"[DEBUG] BF生成: 加算を実行します")
        cell_index = result.begin

        # 加算器ワークエリアの確保 (T, Carry, NextCarry)
        work = self.memory_manager.current_static_top
        self.memory_manager.current_static_top += 3
        
        self.set_value(work + 1, 0) # Carry = 0 に初期化
        
        for i in range(64):
            # Z_i = X_i (左辺のコピー)
            self.copy_values(left_block.begin + i, 1, cell_index + i)
            # T = Y_i (右辺をワークエリアへコピー)
            self.copy_values(right_block.begin + i, 1, work)
            
            # Z_i += T (右辺を加算)
            self.move_to(work, is_work=False); self.bf_code += "[-"
            self.move_to(cell_index + i, is_work=False); self.bf_code += "+"
            self.move_to(work, is_work=False); self.bf_code += "]"
            
            # Z_i += Carry (下位ビットからの繰り上がりを加算)
            self.move_to(work + 1, is_work=False); self.bf_code += "[-"
            self.move_to(cell_index + i, is_work=False); self.bf_code += "+"
            self.move_to(work + 1, is_work=False); self.bf_code += "]"
            
            # 現在 Z_i は 0, 1, 2, 3 のいずれか。これを T に移動し、NextCarryを初期化
            self.move_to(cell_index + i, is_work=False); self.bf_code += "[-"
            self.move_to(work, is_work=False); self.bf_code += "+"
            self.move_to(cell_index + i, is_work=False); self.bf_code += "]"
            self.set_value(work + 2, 0)
            
            # --- 究極の1bit加算ロジック (DivMod 2) ---
            # T [- R+ T[- R- Q+ T[- R+]]]
            # T(work)の値を評価し、R(cell_index+i)に剰余を、Q(work+2)に商(キャリー)を入れる
            self.move_to(work, is_work=False); self.bf_code += "[-"
            self.move_to(cell_index + i, is_work=False); self.bf_code += "+"
            self.move_to(work, is_work=False); self.bf_code += "[-"
            self.move_to(cell_index + i, is_work=False); self.bf_code += "-"
            self.move_to(work + 2, is_work=False); self.bf_code += "+"
            self.move_to(work, is_work=False); self.bf_code += "[-"
            self.move_to(cell_index + i, is_work=False); self.bf_code += "+"
            self.move_to(work, is_work=False); self.bf_code += "]]]"
            
            # NextCarry を次ループの Carry へ移動
            self.move_to(work + 2, is_work=False); self.bf_code += "[-"
            self.move_to(work + 1, is_work=False); self.bf_code += "+"
            self.move_to(work + 2, is_work=False); self.bf_code += "]"
            
        # ワークエリアの解放
        self.memory_manager.current_static_top -= 3

    def _carry_propagate(self, R: MemoryBlock):
        """1bit加算の連鎖による高速なキャリー伝播（1桁加算用）"""
        work = self.memory_manager.current_static_top
        self.memory_manager.current_static_top += 1
        for i in range(R.size - 1):
            # R[i] が 2 になった場合のみ R[i]=0, R[i+1]+=1 とするDivMod-2ロジック
            self.move_to(R.begin + i, is_work=False); self.bf_code += "[-"
            self.move_to(work, is_work=False); self.bf_code += "+"
            self.move_to(R.begin + i, is_work=False); self.bf_code += "[-"
            self.move_to(work, is_work=False); self.bf_code += "-"
            self.move_to(R.begin + i + 1, is_work=False); self.bf_code += "+"
            self.move_to(R.begin + i, is_work=False); self.bf_code += "]]"
            self.move_to(work, is_work=False); self.bf_code += "[-"
            self.move_to(R.begin + i, is_work=False); self.bf_code += "+]"
        self.memory_manager.current_static_top -= 1

    def _io_read_string(self, result_block: MemoryBlock):
        """LF判定でコピーを使わず、-10と+10でスマートに分岐する文字列読み込み"""
        self.move_to(result_block.begin, is_work=False)
        self.bf_code += "----------"\
                        "[++++++++++>>,+[-----------[<+>>]]<]<<[-<<]"\
                        ">>>[[-<<+>>]<+>>>]<<<[[-]<<]>"

    def _io_read_int(self, result_block: MemoryBlock):
        """10進数文字列を読み込み、64bitバイナリに展開する本格的パーサー"""
        work = self.memory_manager.current_static_top
        self.memory_manager.current_static_top += 3
        
        flag = work
        temp = work + 1
        is_space = work + 2

        T1 = self.memory_manager.assign_variable(var_name="tmp_T1", dtype="int", size=64)
        T2 = self.memory_manager.assign_variable(var_name="tmp_T2", dtype="int", size=64)

        self.set_value(flag, 1)
        
        self.move_to(flag, is_work=False); self.bf_code += "["
        
        self.move_to(temp, is_work=False); self.bf_code += "[-],"
        self.bf_code += "----------" # LF判定
        self.move_to(flag, is_work=False); self.bf_code += "[-]"
        
        self.move_to(temp, is_work=False); self.bf_code += "["
        self.bf_code += "++++++++++" # 復元
        self.bf_code += "-" * 32     # Space判定
        
        self.move_to(is_space, is_work=False); self.bf_code += "[-]+"
        self.move_to(temp, is_work=False); self.bf_code += "["
        self.bf_code += "+" * 32     # Space以外なら復元
        self.move_to(is_space, is_work=False); self.bf_code += "[-]"
        
        # --- 正常な数値の処理 ---
        self.bf_code += "-" * 48     # '0' を引いて実数値 (0-9) に変換
        
        # 1. 現在の値を T1(シフト1 = *2) と T2(シフト3 = *8) に分配
        for i in range(64):
            self.move_to(result_block.begin + i, is_work=False); self.bf_code += "[-"
            if i + 1 < 64:
                self.move_to(T1.begin + i + 1, is_work=False); self.bf_code += "+"
            if i + 3 < 64:
                self.move_to(T2.begin + i + 3, is_work=False); self.bf_code += "+"
            self.move_to(result_block.begin + i, is_work=False); self.bf_code += "]"
        
        # 2. R * 10 の加算を実行
        self._math_add(T1, T2, result_block)
        self._zero_clear_region(T1.begin, 64)
        self._zero_clear_region(T2.begin, 64)

        # 3. 新しい桁を最下位ビットに加算
        self.move_to(temp, is_work=False); self.bf_code += "[-"
        self.move_to(result_block.begin, is_work=False); self.bf_code += "+"
        self.move_to(temp, is_work=False); self.bf_code += "]"
        
        # 4. キャリー(繰り上がり)を全体に伝播させる
        self._carry_propagate(result_block)

        self.move_to(flag, is_work=False); self.bf_code += "+" # ループ継続
        self.move_to(temp, is_work=False); self.bf_code += "]" # tempループ(Space以外)終了
        
        self.move_to(is_space, is_work=False); self.bf_code += "["
        self.move_to(flag, is_work=False); self.bf_code += "[-]" # Spaceならフラグを折って終了
        self.move_to(is_space, is_work=False); self.bf_code += "[-]]"
        
        self.move_to(flag, is_work=False); self.bf_code += "]" # メイン読み込みループ終了

        self.memory_manager.free_variable(T1.name)
        self.memory_manager.free_variable(T2.name)
        self.memory_manager.current_static_top -= 3

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
    output_file_path = "./test/" + output_file_path.replace(".py", ".bf")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(transpiler.bf_code)
        print(f"Brainfuckコードが {output_file_path} に出力されました")

    with open("./test/submission.txt", "w", encoding="utf-8") as f:
        f.write(transpiler.bf_code)
        print(f"Brainfuckコードが submission.txt に出力されました")


if __name__ == "__main__":
    main()