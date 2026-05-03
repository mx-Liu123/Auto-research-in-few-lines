# 任务背景与进展

## 任务背景

实现一个轻量级 Python 微框架 `arif`，用于 LLM 驱动的研究实验。核心理念是 **"Like a caveman"** —— 用户显式控制外层循环/逻辑，库只负责"脏活"：工作区隔离、快照、哈希守护、历史记录、CLI 适配。

## 当前进展

**已完成：**
- 包结构重构完成 (`arif/` 包含 `__init__.py`, `agent.py`, `auto_research.py`, `guard.py`, `llm_adapters/`)
- `AutoResearch` 实现了分支管理、实验快照、历史记录、保护文件哈希守护
- `AIAgent` 实现了 CLI 适配器调用和会话管理
- 打包配置完成 (`pyproject.toml`)
- 示例脚本创建完成 (`example/diabetes_sklearn/arif_run.py`)
- 用户已修复 Claude Code 默认模型配置

**待完成：**
- `AIAgent.execute_safe` 需要增强错误检测：检测 CLI 返回的 JSON 中 `is_error: true`，打印错误详情并抛出异常
- 确保模型选择入口正常工作
- 运行完整实验验证整个流程

## 项目结构

```
/home/liumx/momentum_transformer/Auto-research-in-few-lines/
├── arif/
│   ├── __init__.py           # 导出 AutoResearch, AIAgent
│   ├── agent.py              # AIAgent 类，调用 CLI 适配器
│   ├── auto_research.py      # AutoResearch 类，分支/实验管理
│   ├── guard.py              # Guard 类，保护文件哈希守护
│   └── llm_adapters/
│       ├── __init__.py       # get_adapter() 工厂
│       ├── base.py           # BaseAdapter 抽象基类
│       ├── claude.py         # Claude CLI 适配器
│       ├── gemini.py         # Gemini CLI 适配器
│       ├── qwen.py           # Qwen CLI 适配器
│       └── opencode.py       # OpenCode CLI 适配器
├── example/
│   ├── basic_loop.py         # 基础循环示例
│   └── diabetes_sklearn/
│       ├── arif_run.py       # 可运行示例脚本 (project_root="./")
│       ├── evaluator.py      # 评估器 (保护文件)
│       ├── strategy.py       # 策略文件 (LLM 可修改)
│       ├── history.json      # 实验历史记录
│       └── agent_workspaces/ # 实验快照目录
│           └── Branch1/
│               └── exp1.0.0/ # 基线实验
│                   └── history.json
├── pyproject.toml            # 包配置
└── README.md                 # 项目说明
```

## 核心代码待修改位置

**`arif/agent.py` - 需要增加错误检测：**
- `execute_safe` 方法中，`subprocess.run` 后需检测返回 JSON 是否包含 `is_error: true`
- 如果有错误，打印错误详情并抛出异常

**`arif/llm_adapters/claude.py` - 可能需要修改：**
- `parse_output` 方法需确保错误信息能被 `agent.py` 检测到

**运行命令：**
```bash
cd /home/liumx/momentum_transformer/Auto-research-in-few-lines/example/diabetes_sklearn
conda run -n lean python arif_run.py
```

## 关键代码片段

**agent.py 当前结构（需要修改）：**
```python
def execute_safe(self, prompt, guard, new_session=False):
    if new_session:
        self.session_id = None
    guard.before()
    adapter = get_adapter(self.model)
    cmd = adapter.build_command(prompt, self.session_id)
    result = subprocess.run(cmd, **adapter.get_run_kwargs())
    response = adapter.parse_output(result.stdout)
    if response and "session_id" in response:
        self.session_id = response["session_id"]
    guard.after()
    return response.get("text", "") if isinstance(response, dict) else response
```

需要增加：检测 `is_error` 字段，打印错误，抛出异常。
