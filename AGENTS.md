# Codex Testing & Token Efficiency Rules

从现在开始，整个项目遵循“最小必要测试”原则，目标是在保证正确性的前提下减少无关测试、输出和 token 使用。

## 1. 修改代码时只运行必要测试

如果本轮只修改：
```
storage.py
tests/test_storage.py
```

则默认只运行：
```
pytest tests/test_storage.py -q
```

不要自动运行：
```
pytest -q
```

也不要重复运行与本轮改动无关的 schema、agent、tools 测试。

---

## 2. 增量测试优先

新增或修改代码后：

先运行与本次改动直接相关的最小测试集。

例如：

### 修改 schemas.py
```
pytest tests/test_schemas.py -q
```

### 修改 storage.py
```
pytest tests/test_storage.py -q
```

### 修改 research_tools.py
```
pytest tests/test_research_tools.py -q
```

### 修改 research_agent.py
```
pytest tests/test_research_agent.py -q
```

---

## 3. 单个 bug 优先跑单个 test

如果只是在修复某一个失败测试，例如：
```
test_duplicate_source_detection
```

优先运行：
```
pytest tests/test_storage.py::test_duplicate_source_detection -q
```

不要每次修一行代码都跑整个 test file。

修复完成后，再运行当前模块的完整 test file。

---

## 4. 什么时候才运行全量测试

只有以下情况才运行：
```
pytest -q
```

- 修改公共 schema，可能影响多个模块
- 修改跨模块共享接口
- 完成一个开发阶段准备 commit
- 用户明确要求完整 regression
- 当前改动可能影响未知模块

否则默认不运行全量测试。

---

## 5. 不要为了“证明没问题”重复测试

如果同一代码状态已经通过：
```
pytest tests/test_storage.py -q
```

且代码没有再发生变化，不要重复运行同一测试。

---

## 6. 测试数量保持 minimum necessary

新增功能时，只添加能验证：
```
核心正常路径
关键边界
关键失败模式
```

的测试。

不要为了追求高测试数量而：

- 为同一行为写大量等价测试
- 为所有 trivial getter 写独立测试
- 为 Pydantic 自身已经保证的行为写大量重复测试
- 创建几十个参数几乎相同的 case

优先测试我们自己的业务规则。

---

## 7. 输出保持精简

运行 pytest 默认使用：
```
-q
```

除非出现失败需要调试，否则不要使用：
```
-v
-vv
-s
```

如果失败，只展开相关失败测试。

---

## 8. Codex 完成任务后的汇报保持简短

只汇报：

1. 修改了哪些文件
2. 核心实现内容
3. 运行了哪些必要测试
4. 测试结果
5. 是否存在已知限制

不要重复完整实现计划，也不要粘贴大段代码，除非我要求。

---

## 9. 不自动扩大测试范围

不要因为“顺便”而：

- 修改无关测试
- 重构测试框架
- 增加 coverage 工具
- 增加 benchmark
- 增加 CI
- 增加 lint/type-check pipeline

除非当前任务明确需要。

---

## 10. 开发原则

默认流程：
```bash
修改代码
→ 跑最小相关 test
→ 修失败
→ 跑当前模块 test file
→ Stop
```

阶段结束时才：
```sql
相关模块全部通过
→ 必要时 full regression
→ commit
```

核心目标：

> 每次只验证当前改动真正可能影响的行为，不做无意义的全量回归。
