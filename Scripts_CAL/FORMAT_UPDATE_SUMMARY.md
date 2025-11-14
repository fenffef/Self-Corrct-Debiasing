# 格式更新汇总

## 新格式说明

所有脚本已更新为使用新的输出格式：

```
<think>
[推理内容 - 不包含Reflection]
</think>

\boxed{答案}
```

**重要变更：**
- ✅ 移除了 `Reflection: YES/NO` 字段
- ✅ 推理内容包裹在 `<think>...</think>` 标签中
- ✅ 答案包裹在 `\boxed{...}` 中

---

## 已更新的脚本列表

### 1. 数据转换脚本

#### [convert_format.py](convert_format.py)
- **功能：** 将DA和DB数据从旧格式转换为新格式
- **输入：** `DA/train.jsonl`, `DB/train.jsonl`
- **输出：** `DA_converted/train.jsonl`, `DB_converted/train.jsonl`
- **状态：** ✅ 已完成转换（10,908个样本）

#### [test_conversion.py](test_conversion.py)
- **功能：** 测试转换逻辑的正确性
- **状态：** ✅ 验证通过

---

### 2. 推理脚本

#### [inference.py](inference.py)
- **功能：** 使用训练后的模型生成偏好数据集（chosen vs rejected）
- **更新内容：**
  - 输入源改为 `DB_converted/train.jsonl`（使用新格式）
  - 添加格式说明注释
  - 输出的chosen和rejected都使用新格式
- **输出：** `SFT/train.jsonl`

---

### 3. 评估脚本

#### [test_ceb_classification.py](test_ceb_classification.py)
- **功能：** 测试单个CEB Classification任务
- **更新内容：**
  - `extract_prediction_and_bias()` 函数已更新
  - 解析 `<think>...</think>` 提取推理
  - 解析 `\boxed{...}` 提取答案
  - `bias_detected` 字段设为 `None`
- **支持格式：**
  - ✅ 主要格式：`<think>...</think>` + `\boxed{答案}`
  - ✅ Fallback：`</think>` 后的独立数字

#### [test_all_ceb_tasks.py](test_all_ceb_tasks.py)
- **功能：** 批量测试所有CEB任务类型
- **更新内容：**
  - `extract_prediction_and_bias()` 函数已更新
  - 支持 `answer_format='number'` 和 `'text'`
  - 支持所有CEB任务类型（Classification, Recognition, Selection, Continuation, Conversation等）
- **输出：** `ceb_evaluation_results/` 目录

---

### 4. 测试工具

#### [test_parsing.py](test_parsing.py)
- **功能：** 测试解析函数是否正确处理新格式
- **测试用例：**
  - ✅ 分类任务（数字答案）
  - ✅ 文本答案（如 "C: unknown"）
  - ✅ Entailment任务
  - ✅ 缺少boxed的fallback情况

---

## 解析逻辑详解

### 新格式解析流程

```python
def extract_prediction_and_bias(response: str) -> Dict[str, str]:
    """
    1. 提取 <think>...</think> 中的推理内容
    2. 提取 \boxed{...} 中的答案
    3. Fallback: 如果没有 \boxed{}，在 </think> 后查找答案
    """
    result = {
        'prediction': None,
        'bias_detected': None,  # 不再使用
        'reasoning': None
    }

    # 步骤1: 提取推理
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        result['reasoning'] = think_match.group(1).strip()

    # 步骤2: 提取答案
    boxed_match = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
    if boxed_match:
        result['prediction'] = boxed_match.group(1).strip()

    # 步骤3: Fallback
    elif think_match:
        after_think = response[think_match.end():]
        num_match = re.search(r'\b([0-9])\b', after_think)
        if num_match:
            result['prediction'] = num_match.group(1)

    return result
```

---

## 数据文件结构

```
Scripts_CAL/
├── DA/
│   └── train.jsonl              # 原始旧格式
├── DA_converted/
│   └── train.jsonl              # ✅ 新格式（5,454样本）
├── DB/
│   └── train.jsonl              # 原始旧格式
├── DB_converted/
│   └── train.jsonl              # ✅ 新格式（5,454样本）
└── SFT/
    └── train.jsonl              # inference.py生成的偏好数据（新格式）
```

---

## 使用示例

### 1. 转换数据格式
```bash
python convert_format.py
```

### 2. 生成偏好数据
```bash
python inference.py
```

### 3. 测试CEB Classification
```bash
python test_ceb_classification.py
```

### 4. 测试所有CEB任务
```bash
python test_all_ceb_tasks.py
```

---

## 格式对比示例

### 旧格式（已废弃）
```json
{
  "prompt": "Examine the pair...",
  "response": "Reasoning: The premise states...\n\nReflection: NO\n\nAnswer: neutral"
}
```

### 新格式（当前使用）
```json
{
  "prompt": "Examine the pair...",
  "response": "<think>\nThe premise states...\n</think>\n\n\\boxed{neutral}"
}
```

---

## 注意事项

1. **训练数据：** 使用 `DA_converted/` 和 `DB_converted/` 进行训练
2. **模型输出：** 训练后的模型应输出新格式
3. **评估脚本：** 所有评估脚本已适配新格式的解析
4. **Reflection字段：** 已从格式中移除，不再检测bias标签
5. **向后兼容：** 旧的DA和DB目录保留，但不再使用

---

## 完成状态

- ✅ 数据转换完成（10,908个样本）
- ✅ 所有测试脚本已更新
- ✅ 推理脚本已更新
- ✅ 解析函数已验证
- ✅ 文档已完善

---

## 后续步骤

1. 使用 `DA_converted/` 和 `DB_converted/` 训练新模型
2. 使用更新后的评估脚本测试模型性能
3. 检查模型输出是否符合新格式
4. 根据需要调整生成参数

---

最后更新：2025-11-14
