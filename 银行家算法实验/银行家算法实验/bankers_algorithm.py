# -*- coding: utf-8 -*-
"""
银行家算法（Banker's Algorithm）综合实验
=========================================
系统软件综合训练 · 第一次实验

功能概览
--------
1. 随机生成实验数据：客户数 n、资源种类数 m、资源总量 Total、
   最大需求矩阵 Max、已分配矩阵 Allocation、需求矩阵 Need；
2. 安全性检查：判断当前系统状态是否安全，并给出一条安全序列；
3. 查找【所有】安全序列，计算每条序列的“资源利用效率”，
   并按效率从高到低排序输出；
4. 资源请求处理：试分配 + 安全性检查，成功则真正分配，
   失败说明原因（超过最大需求 / 可用资源不足 / 将导致不安全状态）并回滚；
5. Tkinter 图形界面展示系统状态、请求结果、全部安全序列；
   控制台同步输出详细日志。

运行方式
--------
    python bankers_algorithm.py              # 启动图形界面（默认）
    python bankers_algorithm.py --cli        # 控制台演示模式（无图形环境时使用）
    python bankers_algorithm.py --self-test  # 运行内置自测试
    python bankers_algorithm.py -n 6 -m 4 --seed 42   # 指定规模与随机种子

关键定义说明
------------
资源利用效率（依据实验要求“已分配资源占总资源的比例”）：
    按某条安全序列依次执行各进程，记录每个进程开始执行【前】系统的资源
    利用率（已分配资源总量 / 资源总量），整条序列执行过程中各时刻利用率
    的平均值，即为该安全序列的资源利用效率。越早让高占用进程完成并释放
    资源，利用率下降越快，序列的平均利用率也随之不同，因此可用于排序。

需求矩阵 Need = Max - Allocation，天然满足“Need 不超过最大需求 Max”。
随机生成数据后会进行安全性检查，保证初始状态处于安全状态。
"""

import argparse
import itertools
import random
import time

# 枚举安全序列的条数上限（n 较大时全排列可能爆炸，超出后界面/日志会给出提示）
MAX_ENUMERATED_SEQUENCES = 20000

TOTAL_MIN, TOTAL_MAX = 5, 20   # 每类资源总量随机范围（实验建议 5-20 单位）
DEFAULT_N, DEFAULT_M = 5, 3    # 默认客户数 / 资源种类数（建议 3-10 / 3-5）


def fmt_seq(seq):
    """把进程号序列格式化为 P1 → P3 → P4 的形式。"""
    return " → ".join(f"P{p}" for p in seq)


def _matrix_lines(title, mat):
    """把矩阵格式化为对齐的文本行（等宽字体下显示整齐）。"""
    m = len(mat[0])
    w = max(2, max(len(str(v)) for row in mat for v in row))
    lines = [title]
    lines.append("       " + " ".join(f"{'R' + str(j):>{w}}" for j in range(m)))
    for i, row in enumerate(mat):
        lines.append(f"   P{i:<2} " + " ".join(f"{v:>{w}}" for v in row))
    return lines


def _perm_feasible(banker, perm):
    """验证一个进程排列是否为合法安全序列（自测试用）。"""
    work = banker.available[:]
    for pid in perm:
        if any(banker.need[pid][j] > work[j] for j in range(banker.m)):
            return False
        for j in range(banker.m):
            work[j] += banker.allocation[pid][j]
    return True


class Banker:
    """银行家算法核心模型（与界面完全解耦，可单独测试与复用）。"""

    def __init__(self, total, allocation, max_demand):
        self.total = list(total)                          # 每类资源总量
        self.n = len(allocation)                          # 客户（进程）数
        self.m = len(self.total)                          # 资源种类数
        self.allocation = [list(r) for r in allocation]   # 已分配矩阵
        self.max_demand = [list(r) for r in max_demand]   # 最大需求矩阵
        # Need = Max - Allocation（保证 Need 不超过 Max）
        self.need = [[self.max_demand[i][j] - self.allocation[i][j]
                      for j in range(self.m)] for i in range(self.n)]
        # Available = Total - 各进程已分配之和
        self.available = [self.total[j] - sum(self.allocation[i][j]
                                              for i in range(self.n))
                          for j in range(self.m)]

    # ---------------- 数据合法性校验 ----------------
    def validate(self):
        assert all(a >= 0 for a in self.available), "Available 出现负数"
        for i in range(self.n):
            for j in range(self.m):
                assert 0 <= self.allocation[i][j] <= self.max_demand[i][j]
                assert self.need[i][j] == self.max_demand[i][j] - self.allocation[i][j]
        for j in range(self.m):
            assert sum(self.allocation[i][j] for i in range(self.n)) <= self.total[j]

    # ---------------- 随机数据生成（实验要求 1） ----------------
    @classmethod
    def random_instance(cls, n, m, rng=None, log=lambda *a: None, max_attempts=300):
        """按实验要求随机生成一组【保证初始状态安全】的数据。"""
        rng = rng or random.Random()
        total, max_demand = None, None
        for attempt in range(1, max_attempts + 1):
            # 每类资源总量：5-20 随机
            total = [rng.randint(TOTAL_MIN, TOTAL_MAX) for _ in range(m)]
            # 每个客户占用的资源种类数：1-m 随机；占用种类的最大需求：1-资源上限
            max_demand = []
            for _ in range(n):
                kinds = rng.sample(range(m), rng.randint(1, m))
                row = [0] * m
                for j in kinds:
                    row[j] = rng.randint(1, total[j])
                max_demand.append(row)
            # 已分配矩阵：0-Max 随机，且每类资源已分配总量不超过资源总量
            allocation = [[rng.randint(0, max_demand[i][j]) for j in range(m)]
                          for i in range(n)]
            for j in range(m):
                while sum(allocation[i][j] for i in range(n)) > total[j]:
                    i = rng.randrange(n)
                    if allocation[i][j] > 0:
                        allocation[i][j] -= 1
            inst = cls(total, allocation, max_demand)
            safe, _ = inst.is_safe()
            if safe:
                log(f"随机数据生成成功（第 {attempt} 次尝试即得到安全初始状态）")
                return inst
        # 兜底：全零分配必为安全状态
        log("多次尝试未生成安全状态，退回全零分配（必安全）")
        return cls(total, [[0] * m for _ in range(n)], max_demand)

    # ---------------- 安全性检查（实验要求 2 基础） ----------------
    def is_safe(self):
        """标准安全性算法：返回 (是否安全, 一条安全序列)。"""
        work = self.available[:]
        finish = [False] * self.n
        seq = []
        while len(seq) < self.n:
            progressed = False
            for i in range(self.n):
                if not finish[i] and all(self.need[i][j] <= work[j]
                                         for j in range(self.m)):
                    for j in range(self.m):
                        work[j] += self.allocation[i][j]
                    finish[i] = True
                    seq.append(i)
                    progressed = True
            if not progressed:
                break
        return len(seq) == self.n, seq

    # ---------------- 查找所有安全序列（实验要求 2） ----------------
    def find_all_safe_sequences(self, cap=MAX_ENUMERATED_SEQUENCES):
        """DFS 枚举所有安全序列；返回 (序列列表, 是否因达到上限被截断)。"""
        results = []
        work = self.available[:]
        finish = [False] * self.n
        path = []
        truncated = False

        def dfs():
            nonlocal truncated
            if truncated:
                return
            if len(path) == self.n:
                results.append(tuple(path))
                if len(results) >= cap:
                    truncated = True
                return
            for i in range(self.n):
                if truncated:
                    return
                if not finish[i] and all(self.need[i][j] <= work[j]
                                         for j in range(self.m)):
                    finish[i] = True
                    path.append(i)
                    for j in range(self.m):
                        work[j] += self.allocation[i][j]
                    dfs()
                    for j in range(self.m):
                        work[j] -= self.allocation[i][j]
                    path.pop()
                    finish[i] = False

        dfs()
        return results, truncated

    # ---------------- 资源利用效率（实验要求 2） ----------------
    def sequence_efficiency(self, seq):
        """计算一条安全序列的资源利用效率（定义见文件头注释）。"""
        work = self.available[:]
        tot = sum(self.total)
        utils = []
        for pid in seq:
            allocated = sum(self.total[j] - work[j] for j in range(self.m))
            utils.append(allocated / tot)
            for j in range(self.m):
                work[j] += self.allocation[pid][j]
        return sum(utils) / len(utils) if utils else 0.0

    def sorted_sequences(self, cap=MAX_ENUMERATED_SEQUENCES):
        """所有安全序列 + 资源利用效率，按效率从高到低排序。"""
        seqs, truncated = self.find_all_safe_sequences(cap)
        scored = [(s, self.sequence_efficiency(s)) for s in seqs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored, truncated

    def utilization(self):
        """当前系统资源利用率 = 已分配资源总量 / 资源总量。"""
        tot = sum(self.total)
        return sum(sum(r) for r in self.allocation) / tot if tot else 0.0

    # ---------------- 资源请求处理（实验要求 3） ----------------
    def request(self, pid, request, log=lambda *a: None):
        """
        处理进程 pid 的资源请求。
        返回 (是否成功, 结果说明, 成功时的一条安全序列或 None)。
        失败时说明原因；试分配后若不安全则回滚，系统状态不变。
        """
        request = list(request)
        log(f"[请求] P{pid} 请求资源 {request}")
        log(f"       当前 Available={self.available}，P{pid} 的 Need={self.need[pid] if 0 <= pid < self.n else '?'}")
        if not (0 <= pid < self.n):
            msg = f"失败：进程号 P{pid} 不存在"
            log("       " + msg)
            return False, msg, None
        if any(r < 0 for r in request) or len(request) != self.m:
            msg = f"失败：请求向量不合法（需为 {self.m} 个非负整数）"
            log("       " + msg)
            return False, msg, None
        # 第 1 步：请求不能超过其剩余需求
        if any(request[j] > self.need[pid][j] for j in range(self.m)):
            msg = f"失败：请求量超过 P{pid} 的剩余需求 Need={self.need[pid]}，请求非法"
            log("       " + msg)
            return False, msg, None
        # 第 2 步：请求不能超过系统可用资源
        if any(request[j] > self.available[j] for j in range(self.m)):
            msg = f"失败：系统可用资源不足（Available={self.available}），P{pid} 需等待"
            log("       " + msg)
            return False, msg, None
        # 第 3 步：试分配
        old_available = self.available[:]
        old_alloc_row = self.allocation[pid][:]
        old_need_row = self.need[pid][:]
        for j in range(self.m):
            self.available[j] -= request[j]
            self.allocation[pid][j] += request[j]
            self.need[pid][j] -= request[j]
        log(f"       试分配后 Available={self.available}，进行安全性检查……")
        # 第 4 步：安全性检查，安全则确认分配，不安全则回滚
        safe, seq = self.is_safe()
        if safe:
            msg = f"成功：分配后系统仍处于安全状态，安全序列之一：{fmt_seq(seq)}"
            log("       " + msg)
            return True, msg, seq
        self.available = old_available
        self.allocation[pid] = old_alloc_row
        self.need[pid] = old_need_row
        msg = "失败：若分配，系统将进入不安全状态（可能发生死锁），已回滚并拒绝请求"
        log("       " + msg)
        return False, msg, None

    # ---------------- 状态文本（界面 / 控制台共用） ----------------
    def state_text(self):
        lines = [
            f"资源总量 Total    : {self.total}",
            f"可用资源 Available: {self.available}",
            f"当前资源利用率（已分配/总量）: {self.utilization():.2%}",
            "",
        ]
        lines += _matrix_lines("已分配矩阵 Allocation:", self.allocation)
        lines.append("")
        lines += _matrix_lines("最大需求矩阵 Max:", self.max_demand)
        lines.append("")
        lines += _matrix_lines("需求矩阵 Need (= Max - Allocation):", self.need)
        return "\n".join(lines)


# =====================================================================
# 内置自测试：经典教材示例 + 随机实例批量验证
# =====================================================================
def run_self_test():
    print("=" * 64)
    print("银行家算法 · 内置自测试")
    print("=" * 64)

    # ---- 1) 经典教材示例（Silberschatz《操作系统概念》） ----
    total = [10, 5, 7]
    allocation = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    max_demand = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    b = Banker(total, allocation, max_demand)
    b.validate()
    assert b.available == [3, 3, 2] and b.need[0] == [7, 4, 3]
    safe, seq = b.is_safe()
    assert safe, "经典示例应为安全状态"
    print(f"1. 经典示例安全性检查通过，安全序列之一：{fmt_seq(seq)}")

    all_seqs, truncated = b.find_all_safe_sequences()
    brute = [p for p in itertools.permutations(range(5)) if _perm_feasible(b, p)]
    assert not truncated and len(all_seqs) == len(brute), "枚举结果应与全排列验证一致"
    print(f"2. 安全序列枚举正确：共 {len(all_seqs)} 条，与全排列暴力验证一致")

    ok, msg, _ = b.request(1, [1, 0, 2], log=print)
    assert ok, "P1 请求 (1,0,2) 应成功"
    ok, msg, _ = b.request(4, [3, 3, 0], log=print)
    assert not ok and "不足" in msg, "P4 请求 (3,3,0) 应因资源不足失败"
    ok, msg, _ = b.request(0, [0, 2, 0], log=print)
    assert not ok and "不安全" in msg, "P0 请求 (0,2,0) 应因不安全被拒绝"
    print("3. 经典示例三类请求（成功 / 资源不足 / 不安全回滚）全部符合预期")

    # ---- 2) 随机实例批量验证 ----
    rng = random.Random(2024)
    for t in range(30):
        n, m = rng.randint(3, 8), rng.randint(3, 5)
        inst = Banker.random_instance(n, m, rng)
        inst.validate()                       # 数据合法性
        safe, _ = inst.is_safe()              # 初始状态必安全
        assert safe
        seqs, trunc = inst.find_all_safe_sequences()
        for s in seqs:                        # 每条枚举序列确实可行
            assert _perm_feasible(inst, s)
        if n <= 6 and not trunc:              # 小规模与全排列结果一致
            brute = [p for p in itertools.permutations(range(n))
                     if _perm_feasible(inst, p)]
            assert len(brute) == len(seqs)
        scored, _ = inst.sorted_sequences()   # 效率值域 [0,1] 且降序
        effs = [e for _, e in scored]
        assert effs == sorted(effs, reverse=True)
        assert all(0.0 <= e <= 1.0 for e in effs)
    print("4. 30 组随机实例校验通过（数据合法性 / 初始安全性 / 序列有效性 / 效率排序）")

    print("\n全部自测试通过 ✅")


# =====================================================================
# 控制台演示模式
# =====================================================================
def run_cli(args):
    if not (1 <= args.n <= 50 and 1 <= args.m <= 10):
        print("参数错误：n 建议 3-10，m 建议 3-5（本程序允许 1<=n<=50, 1<=m<=10）")
        return
    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    print("=" * 64)
    print(f"银行家算法 · 控制台演示（n={args.n}, m={args.m}, seed={args.seed}）")
    print("=" * 64)
    banker = Banker.random_instance(args.n, args.m, rng, log=print)

    print("\n【当前系统状态】")
    print(banker.state_text())

    print("\n【所有安全序列（按资源利用效率从高到低排序）】")
    t0 = time.time()
    scored, truncated = banker.sorted_sequences()
    dt = (time.time() - t0) * 1000
    print(f"共找到 {len(scored)}{'+' if truncated else ''} 条安全序列（耗时 {dt:.1f} ms）：")
    for idx, (s, eff) in enumerate(scored, 1):
        print(f"  {idx:>3}. {fmt_seq(s):<28} 资源利用效率: {eff:.2%}")
    if truncated:
        print(f"  …… 序列过多，仅枚举前 {MAX_ENUMERATED_SEQUENCES} 条")

    print("\n【资源请求演示】")
    banker.request(0, [0] * args.m, log=print)                      # 零请求：必成功
    pid = rng.randrange(args.n)                                     # 随机合法请求
    req = [rng.randint(0, banker.need[pid][j]) for j in range(args.m)]
    banker.request(pid, req, log=print)
    banker.request(0, [x + 1 for x in banker.need[0]], log=print)   # 非法请求：超过 Need

    print("\n【请求后的系统状态】")
    print(banker.state_text())


# =====================================================================
# Tkinter 图形界面（实验要求 3）
# =====================================================================
def launch_gui():
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    class BankerGUI(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("银行家算法模拟系统 · 系统软件综合训练")
            self.geometry("1000x900")
            self.banker = None
            self.rng = random.Random()
            self._build()

        def _build(self):
            # ① 初始数据区
            top = ttk.LabelFrame(self, text="① 初始数据（随机生成）")
            top.pack(fill="x", padx=8, pady=4)
            ttk.Label(top, text="客户数 n：").pack(side="left", padx=(8, 0))
            self.n_var = tk.StringVar(value=str(DEFAULT_N))
            ttk.Entry(top, textvariable=self.n_var, width=5).pack(side="left")
            ttk.Label(top, text="资源种类数 m：").pack(side="left", padx=(12, 0))
            self.m_var = tk.StringVar(value=str(DEFAULT_M))
            ttk.Entry(top, textvariable=self.m_var, width=5).pack(side="left")
            ttk.Button(top, text="🎲 随机生成 / 重新生成",
                       command=self.generate).pack(side="left", padx=12)
            ttk.Label(top, text="（资源总量、Allocation、Max、Need 均自动生成，"
                                "并保证初始状态安全）").pack(side="left", padx=4)

            # ② 系统状态区
            mid = ttk.LabelFrame(self, text="② 当前系统状态（Available / Allocation / Need）")
            mid.pack(fill="both", expand=True, padx=8, pady=4)
            self.state_txt = scrolledtext.ScrolledText(
                mid, height=18, font=("Courier New", 10))
            self.state_txt.pack(fill="both", expand=True, padx=4, pady=4)

            # ③ 资源请求区
            req = ttk.LabelFrame(self, text="③ 资源请求（动态输入）")
            req.pack(fill="x", padx=8, pady=4)
            ttk.Label(req, text="进程：P").pack(side="left", padx=(8, 0))
            self.pid_var = tk.StringVar(value="0")
            ttk.Entry(req, textvariable=self.pid_var, width=5).pack(side="left")
            ttk.Label(req, text="请求向量（空格分隔）：").pack(side="left", padx=(12, 0))
            self.req_var = tk.StringVar()
            ttk.Entry(req, textvariable=self.req_var, width=22).pack(side="left")
            ttk.Button(req, text="提交请求",
                       command=self.submit_request).pack(side="left", padx=12)
            self.result_var = tk.StringVar(value="")
            self.result_lbl = tk.Label(req, textvariable=self.result_var,
                                       anchor="w", justify="left", wraplength=360)
            self.result_lbl.pack(side="left", padx=8, fill="x", expand=True)

            # ④ 安全序列区
            seqf = ttk.LabelFrame(self, text="④ 所有安全序列（按资源利用效率从高到低排序）")
            seqf.pack(fill="both", expand=True, padx=8, pady=4)
            ttk.Button(seqf, text="查找所有安全序列",
                       command=self.show_sequences).pack(anchor="w", padx=4, pady=2)
            self.seq_txt = scrolledtext.ScrolledText(
                seqf, height=10, font=("Courier New", 10))
            self.seq_txt.pack(fill="both", expand=True, padx=4, pady=4)

            # ⑤ 日志区
            logf = ttk.LabelFrame(self, text="⑤ 运行日志（同步输出到控制台）")
            logf.pack(fill="both", expand=True, padx=8, pady=4)
            self.log_txt = scrolledtext.ScrolledText(
                logf, height=8, font=("Courier New", 9))
            self.log_txt.pack(fill="both", expand=True, padx=4, pady=4)

        # 日志：同时写界面日志框与控制台
        def log(self, msg=""):
            self.log_txt.insert("end", msg + "\n")
            self.log_txt.see("end")
            print(msg)

        def generate(self):
            try:
                n, m = int(self.n_var.get()), int(self.m_var.get())
                if not (1 <= n <= 50 and 1 <= m <= 10):
                    raise ValueError
            except ValueError:
                messagebox.showerror("输入错误",
                                     "请输入合法的 n（建议 3-10）和 m（建议 3-5）")
                return
            self.log("\n" + "=" * 58)
            self.log(f"随机生成数据：n={n}, m={m}")
            self.banker = Banker.random_instance(n, m, self.rng, log=self.log)
            self.refresh_state()
            self.seq_txt.delete("1.0", "end")
            self.result_var.set("")
            safe, seq = self.banker.is_safe()
            self.log(f"初始状态安全性检查：{'安全' if safe else '不安全'}"
                     f"，安全序列之一：{fmt_seq(seq)}")
            self.log(f"当前资源利用率（已分配/总量）："
                     f"{self.banker.utilization():.2%}")

        def refresh_state(self):
            self.state_txt.delete("1.0", "end")
            self.state_txt.insert("end", self.banker.state_text())

        def submit_request(self):
            if self.banker is None:
                messagebox.showwarning("提示", "请先点击“随机生成 / 重新生成”创建数据")
                return
            try:
                pid = int(self.pid_var.get())
                req = [int(x) for x in
                       self.req_var.get().replace("，", " ").replace(",", " ").split()]
                if len(req) != self.banker.m:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "输入错误", f"请输入合法的进程号和 {self.banker.m} 维请求向量"
                                f"（如：{' '.join(['1', '0', '2'][:self.banker.m])}）")
                return
            ok, msg, _seq = self.banker.request(pid, req, log=self.log)
            self.result_var.set(msg)
            self.result_lbl.config(fg=("#087f23" if ok else "#c62828"))
            if ok:
                self.refresh_state()
            self.log("")

        def show_sequences(self):
            if self.banker is None:
                messagebox.showwarning("提示", "请先点击“随机生成 / 重新生成”创建数据")
                return
            self.log("开始枚举所有安全序列……")
            t0 = time.time()
            scored, truncated = self.banker.sorted_sequences()
            dt = (time.time() - t0) * 1000
            lines = [f"共找到 {len(scored)}{'+' if truncated else ''} 条安全序列"
                     f"（耗时 {dt:.1f} ms，已按资源利用效率从高到低排序）", ""]
            for idx, (s, eff) in enumerate(scored, 1):
                lines.append(f"{idx:>4}. {fmt_seq(s):<28} 效率: {eff:.2%}")
            if truncated:
                lines.append(f"…… 序列过多，仅枚举前 {MAX_ENUMERATED_SEQUENCES} 条")
            self.seq_txt.delete("1.0", "end")
            self.seq_txt.insert("end", "\n".join(lines))
            self.log(f"安全序列枚举完成：{len(scored)} 条，耗时 {dt:.1f} ms")

    app = BankerGUI()
    app.generate()   # 启动即自动生成一组演示数据
    app.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="银行家算法综合实验程序")
    parser.add_argument("--cli", action="store_true", help="控制台演示模式")
    parser.add_argument("--self-test", action="store_true", help="运行内置自测试")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（便于复现）")
    parser.add_argument("-n", type=int, default=DEFAULT_N, help="客户数（默认 5）")
    parser.add_argument("-m", type=int, default=DEFAULT_M, help="资源种类数（默认 3）")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return
    if args.cli:
        run_cli(args)
        return
    try:
        launch_gui()
    except ImportError:
        print("当前环境未安装 tkinter，自动切换到控制台演示模式（--cli）。\n")
        run_cli(args)


if __name__ == "__main__":
    main()
