# sva-tools 核心架构重构技术方案：从启发式模拟转向形式化见证合成

## 1. 概述

当前 `sva-tools` 在从 SystemVerilog 断言（SVA）提取定时图以及从定时图发射 SVA 时，过度依赖基于有向无环图（DAG）的启发式模拟和简单的宏替换。这种架构导致了多套算法管线并存、错误降级逻辑混乱，且无法处理工业级协议中的多比特向量和稳定性约束。

**核心重构目标**：
*   **语义精确性**：废除"尽力而为"的猜测算法，确保波形图是 SVA 语义的精确呈现。当语义无法精确表达时，必须显式标记为 LOSSY 或 UNSUPPORTED，而非静默降级。
*   **分层求解 (Layered Solving)**：以内置规范化轨迹合成（Canonical Trace Synthesis）为默认路径，以形式化见证合成（Witness Synthesis）为可选的高置信度后端。两层之间通过统一的 `ScenarioDocument` IR 交互。
*   **符号化反向映射**：在见证轨迹中自动识别具体数值并将其映射回原始符号宏（如 `RQ`, `RS`）。
*   **Emit 路径语义完整性**：修复当前 Emit 路径完全忽略 `LaneConstraint`（如 `show high(X) from A until B`）的核心缺陷，使其生成的 SVA 包含区间约束（`until_with`, `$stable`），而非仅有点对点延迟。

---

## 2. 冗余清理清单 (The Kill List)

为消除复杂性，必须移除或重构以下模块：

1.  **启发式波形合成 (`_satisfy_anchor`, `_solve_scenario_ticks`)**：
    *   **位置**：`src/sva_toolkit/timing/projection/wavedrom_view.py`
    *   **理由**：这些算法仅能满足孤立的锚点条件，完全忽略了跨度语义（Span Semantics），是生成错误波形的根源。
    *   **替代**：由规范化轨迹合成（Phase 4.1）接管。
2.  **符号化 SVG 渲染器 (`render/svg.py`)**：
    *   **理由**：工具不应维护多种渲染路径。应统一为"基于严格轨迹的 WaveDrom 渲染"。
3.  **玩具级时序求解器 (`CEGSolver`) —— 重构而非删除**：
    *   **位置**：`src/sva_toolkit/timing/bridge/solver.py`
    *   **理由**：其当前的最长路径算法过于简单，无法处理逻辑蕴含和多通道并发约束。但完全删除会导致在没有外部形式化工具时无法使用。
    *   **替代**：重构为轻量级约束求解器，处理 80% 的常见场景（线性序列、简单分支、基本 hold 约束）。对超出能力范围的约束，显式报错并提示用户启用形式化后端。
4.  **静默降级逻辑**：
    *   **理由**：移除所有吞掉错误并切换到不理想方案的 `try...except` 块。任何无法满足约束的情况必须 Fail-Fast 并报错。

---

## 3. 核心组件增强

### 3.1 Emit 路径的核心缺陷修复 (`emit_sva.py`)

**当前最严重的语义缺陷**：`DiagramCompiler` 仅将 `anchors` 和 `windows` 编译为 CEG，完全忽略 `lane_constraints`。这意味着：

```
TD:   show high(AWVALID) from aw_valid_rise until aw_handshake;
当前: $rose(AWVALID) |-> ##[0:AW_READY_MAX] (AWVALID && AWREADY)  // 缺少区间保持
期望: $rose(AWVALID) |-> AWVALID until_with AWREADY
```

**必须修复的映射规则**：

| TD LaneConstraint | SVA 生成目标 |
|---|---|
| `show high(S) from A until B` | `A \|-> S until_with B` |
| `show low(S) from A until B` | `A \|-> !S until_with B` |
| `show stable(S) from A until B` | `A \|-> $stable(S) until_with B` |
| `show eq(S, VAL) from A until B` | `A \|-> (S == VAL) until_with B` |

不修复此缺陷，后续的见证合成 Harness 也会遗漏区间约束，导致生成的轨迹不完整。

### 3.2 SVA 提取引擎 (`from_sva.py`)
*   **增强点**：在解析 SVA 时，必须精确提取信号的位宽及关联的符号化宏名称。即使 SVA 中仅表现为 `$stable(data)`，也要记录 `data` 的向量属性及其绑定的符号标识。
*   **Bus 推断修复**：当前仅在 `==`/`!=` 比较时推断为 bus。`$stable(signal)` 引用的信号也应标记为 `kind=BUS`（宽度可未知）。

### 3.3 WaveDrom 投影层 (`wavedrom_view.py`)
*   **增强点**：将其从"启发式合成器"降级为"规范化轨迹适配器"。它不再承担基于锚点猜测的合成职责，仅负责将包含完整采样数据（samples）的 `ScenarioDocument` 转换为 WaveDrom JSON。
*   **保留轻量级 tick 分配**：对于已包含 `absolute_tick` 的锚点，直接使用；对于缺少 tick 的文档，由上游的规范化轨迹合成器（§4.1）负责填充，而非由投影层猜测。

### 3.4 参数化发射引擎 (`emit_sva.py`)
*   **增强点**：生成的 SVA 必须是参数化的。时序图中的 `param` 声明应直接映射为 SVA `property` 的输入参数。

---

## 4. 分层求解架构

### 4.1 第一层：规范化轨迹合成（Canonical Trace Synthesis）

**定位**：默认路径，零外部依赖，处理常见的单时钟线性/分支场景。

**算法逻辑**：
1.  从 `ScenarioDocument` 的 anchors/windows 构建约束图。
2.  基于拓扑排序分配 tick 位置（触发点取最早合法 tick，响应窗口取最早合法响应 tick）。
3.  **关键改进**：遍历 `lane_constraints`，在分配的 tick 区间内强制施加跨度语义：
    - `high(S) from A until B` → S 在 [tick_A, tick_B] 内全部为 1
    - `stable(S) from A until B` → S 在 [tick_A, tick_B] 内保持确定性占位值
    - `low(S) from A until B` → S 在 [tick_A, tick_B] 内全部为 0
4.  对无约束区间的 bus lane 填充 `x`，对无约束的 bit lane 填充 `0`。
5.  当约束冲突（如同一 tick 要求 `high(S)` 和 `low(S)`）时，**Fail-Fast 报错**，不做静默降级。

**能力边界**：
- 支持：线性序列、简单分支（AND/OR）、基本 hold/stable 约束、参数化延迟。
- 不支持：`intersect` 精确长度匹配、`first_match` 最短路径选取、多时钟域、本地变量数据流。超出能力时显式报 `UNSUPPORTED`。

### 4.2 第二层：形式化见证合成（Witness Synthesis）

**定位**：可选的高置信度后端，用于工业级签收场景或内置求解器无法处理的复杂属性。

#### Formal Harness 生成
*   **功能**：基于 `ScenarioDocument` 元数据，生成包含信号声明、稳定性假设和 `cover` 属性的 SystemVerilog 测试平台。
*   **处理多比特向量**：在 Harness 中声明正确宽度的位向量（Bit-vectors）。
*   **处理符号参数**：将宏参数声明为输入 Wire，并施加 `$stable` 约束使其成为符号常量。
*   **包含区间约束**：从 `lane_constraints` 生成对应的 `assume`/`cover` 属性，确保 harness 语义完整。
    ```systemverilog
    input wire [31:0] data;
    input wire [31:0] RQ; // 符号化参数宏
    assume property (@(posedge clk) $stable(RQ));
    cover property (@(posedge clk)
        $rose(trigger) |-> (data == RQ) until_with done
    );
    ```

#### 符号值反向映射 (Symbolic Back-Mapping)
*   **算法逻辑**：
    1.  形式化工具会为 `RQ` 分配一个具体的数值（如 `32'hA1`）。
    2.  解析生成的轨迹，识别出求解器为 `RQ` 挑选的具体常量。
    3.  在所有信号采样中，将该具体常量（`32'hA1`）替换回原始的字符串标识符（`"RQ"`）。
    4.  这确保了即使没有具体位宽，时序图也能显示正确的宏名称。

#### 适用的形式化工具
*   EBMC、SymbiYosys（cover 模式）或其他支持 VCD/witness 输出的形式化引擎。
*   见证轨迹需经过确定性后处理（deterministic normalization），避免不同求解器版本产生的 diff 噪声。

---

## 5. 验证策略

### 5.1 Extract 路径 (SVA -> Diagram)
1.  提取 SVA 时保留信号位宽和关联宏。
2.  通过规范化轨迹合成（默认）或见证合成（可选）生成合法轨迹。
3.  执行符号反向映射（仅见证合成路径）。
4.  WaveDrom 渲染。

### 5.2 Emit 路径 (Diagram -> SVA)
1.  将图中的 `from...until` 区域、稳定性约束直接映射为 SVA 的 `until_with` 和 `$stable`。
2.  将图中的参数生成为 SVA 的 `property` 形式参数。

### 5.3 回归验证方向：sv -> td -> sv（而非 td -> sv -> td）

**原文档提出的 td -> sv -> gen_td 闭环验证不可行**，原因如下：
- Emit 路径存在不可逆的语义压缩（lane_constraints 丢失、无界延迟截断、多窗口合并）。
- Extract 路径对部分 SVA 构造标记为 LOSSY/UNSUPPORTED。
- 因此 gen_td 必然是原始 td 的退化子集，比对结果要么永远报 diff（无意义），要么降低比对标准（自欺欺人）。

**正确的回归验证方向是 sv -> td -> sv**：
1.  从 SVA 提取 `ScenarioDocument`。
2.  从 `ScenarioDocument` 重新发射 SVA。
3.  比较原始 SVA 与重新发射的 SVA 的语义等价性。

**理由**：SVA 是形式化语言，两个 SVA 表达式的等价性可通过规范化文本比对或 AST 结构比对来判定，远比两个时序图的"视觉等价性"容易定义和自动化。

**等价性判定方法**：
- **AST 规范化比对**：将两个 SVA 解析为 AST，规范化排序和命名后进行结构比对。
- **形式化等价检查**（可选）：通过 `assert property (p1 iff p2)` 让形式化工具验证语义等价。

**对于 `extraction_status` 为 LOSSY 的属性**，回归验证应检查重新发射的 SVA 是否为原始 SVA 的保守近似（即不会放过原始属性能捕获的违规），而非要求严格等价。

---

## 6. 与现有修复计划的关系

本方案与 `timing_render_extraction_remediation_plan.md` 的分阶段修复计划互补而非替代：

| 修复计划阶段 | 本方案对应 | 关系 |
|---|---|---|
| Phase 1: 修复 bus-kind 推断 | §3.2 | 前置依赖，应先完成 |
| Phase 2: WaveDrom 规则恢复 | §3.3 + §4.1 | 本方案的规范化合成替代当前的启发式恢复 |
| Phase 3: 语义感知的采样合成 | §4.1 | 直接对应 |
| Phase 4: DSL 元数据保留 | 不在本方案范围 | 仍应执行，与本方案正交 |
| Phase 5: 可选见证后端 | §4.2 | 直接对应 |

**建议执行顺序**：
1.  修复 bus-kind 推断（最小改动、最高信号量）
2.  修复 Emit 路径的 lane_constraints 丢失（§3.1，核心语义缺陷）
3.  实现规范化轨迹合成（§4.1，替代启发式合成）
4.  清理冗余模块（§2）
5.  实现 sv -> td -> sv 回归验证（§5.3）
6.  可选：接入形式化见证后端（§4.2）

---

## 7. 结论

重构应致力于将 `sva-tools` 从一个"猜测工具"转变为一个"形式化语义的视觉序列化工具"。但这一转变必须务实：

1.  **先修内伤**：Emit 路径丢失 `lane_constraints` 是当前最大的语义缺陷，优先级高于引入新的形式化后端。
2.  **分层而非替代**：内置规范化求解覆盖常见场景，形式化后端作为可选增强，而非必要依赖。
3.  **验证方向正确**：回归验证走 sv -> td -> sv 路径，利用 SVA 的形式化可比性，而非追求不可能的时序图闭环。
4.  **显式降级**：对无法精确处理的构造标记 LOSSY/UNSUPPORTED，而非静默产出错误结果。
