# 研证AI · EvidenceLab

**科研文献证据分析与结论核验工作台**，武汉1024 AI黑客松参赛公开候选源码。

版本：`0.1.0-public-candidate.20261019`。本版本使用全新提交历史，不包含研究数据库、真实论文或私人模型响应。

EvidenceLab 将原文证据、实验字段、人工审阅、条件比较与报告快照连接起来，让使用者能从结论回到对应证据，检查实验属于哪个方法、采用什么协议，以及哪些条件尚不明确。

## 核心实现

- **证据追溯**：PyMuPDF 提取文本、页码和坐标；原始文件校验、分页预览和高亮帮助核查来源。
- **实验字段审阅**：规则候选、模型候选与人工候选进入审阅流程；方法、数据集、划分、指标、数值、单位和条件绑定引文，采纳后保留修改历史。
- **条件感知比较**：按任务、数据集、划分、方法、协议和指标等条件对齐；证据不充分时保留差异和未知项，不把数值高低直接当作科学结论。
- **可复核报告**：保存比较快照，导出 Markdown/JSON，保留当时的证据、条件和判断；旧快照不自动成为新实验的结论。
- **问答与核验**：本地检索展示来源；可选模型问答和结论核验检查引文。默认本地核验返回证据不足，需要人工复核。

上述字段审阅、条件规则、证据绑定和快照流程是本项目的应用实现。PDF 解析、RAG、语言模型和状态图编排是既有技术；本项目实际使用 PyMuPDF、HTTPX、LangGraph、FastAPI、React 等组件，详见[架构说明](docs/ARCHITECTURE.md)与[第三方声明](THIRD_PARTY_NOTICES.md)。

## 安装和启动

需要 Python 3.11+、Node.js 20.19+ 或 22.12+、npm；已验证环境是 Windows/Python 3.13。尚未完成全新电脑安装测试。依赖需要从软件源下载。

Windows 在项目根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Install
```

之后可双击 `start.bat`。默认地址是 `http://127.0.0.1:8000`；已有服务占用端口时用 `start.ps1 -Port 8790`，不要停止其他项目服务。

手动安装与启动（Windows）：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
cd frontend
npm ci
npm run build
cd ..
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS 可使用 `.venv/bin/python` 执行相同命令。`requirements.txt` 是跨平台直接依赖；`requirements-lock.windows-py313.txt` 是本次 Windows 核心及测试依赖版本闭包，其他平台不应盲目套用。

无模型也能载入原创合成样例，完成证据查看、候选审阅、条件比较和报告导出，见[操作手册](docs/DEMO_RUNBOOK.md)。

## 统一比赛演示（8792）

先将已有、经过原文与 SHA256 核查的研究库备份恢复到一个**新的数据目录**。源码不附带真实论文或研究数据。演示脚本拒绝缺失或空白数据库，不会建立空库替代你的研究记录。

```powershell
.\START_DEMO_AI.ps1 -Python .\.venv\Scripts\python.exe -ConfigFile C:\private\evidencelab.env -DataDir .\artifacts\demo-data
```

本机已有配置文件可通过 `EVIDENCELAB_CONFIG_FILE` 指定；文件只在后端进程内读取，不复制到源码。配置包括 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`，可选 `LLM_TIMEOUT=90`。缺少必需项时列出变量名并保留本地模式。也可在未指定配置文件时使用进程环境变量。`EVIDENCELAB_PYTHON` 可指向已有 Python 环境。

打开 `http://127.0.0.1:8792`，无需切换地址即可查看 PDF、问答、核验、人工实验、条件比较及报告。配置模型后，问答和核验默认选择模型模式；只有主动点击发送才调用模型，加载页面、读取状态与切换菜单不会自动付费。可手动选择本地模式。来源校验、超时、网络错误处理、有限纠错和停止等待操作保留。停止等待不保证服务商立刻中止已经开始的计算或计费。

离线备用：停止已知且空闲的 8792 服务后，执行 `START_DEMO.ps1`（同样可传 `-Python` / `-DataDir`）。它对同一数据副本禁用模型网络。两个脚本均不自动停止占用端口的服务。

可选 `-AuditDir .\artifacts\acceptance-session` 用于私有验收采集，逐次记录用量、正文和结果，不设置人为调用/字符/费用上限，也不自动重试；必须使用新的目录，禁止上传这些材料。费用仅对已知费率的配置模型估算，最终以服务商账单为准。正常演示不需要该参数。

五分钟点击路线与讲稿见 [正式比赛演示](docs/COMPETITION_DEMO.md)。工程与真实验收记录见 [本轮整理记录](docs/UNIFIED_DEMO_ACCEPTANCE.md)。复杂表格与多条件问答仍可能失败，不能把两条文本案例成功推广为全部科研任务可靠。

## 配置和数据

需要模型功能时，将 `.env.example` 复制为本机 `.env`，填写 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。密钥只由后端读取；不要上传 `.env`。模型请求可能产生服务商费用，并会向配置服务发送所选原文。真实调用结果与失败必须按本次请求记录，不能用旧响应冒充新成功。

默认使用本项目 `data/`。移动项目时一起保留数据库和 `data/files`，路径解析仍限制在数据目录内。备份用法：

```powershell
.\.venv\Scripts\python scripts/backup_library.py --help
```

备份必须放在新的独立目录。数据库、PDF、备份、日志及原始请求响应不得提交到 Git。本地诊断内容采集默认关闭，只允许写入被忽略的 `artifacts/`。

## 验证

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts/smoke_local.py
cd frontend
npm test
npm run build
```

本版本实际测试数量、隔离验证、清理范围和发布条件见[公开候选验收记录](docs/PUBLIC_CANDIDATE_REVIEW.md)。测试采用原创合成材料和模拟模型响应；工程测试通过不代表真实科研自动提取已可靠。

## 能力边界

复杂表格、跨句归属及完整多问题回答仍可能失败。模型候选须逐项原文审阅；安全拒绝不等于模型提取成功。支持/反驳/不足标签不能替代科学判断。

PDF 要求可提取文字层，没有 OCR。规则比较基于已报告且已审阅的字段，不进行统计显著性推断。本地服务没有多用户登录体系，默认仅监听回环地址，不能直接用作开放互联网服务。

## 开源许可

Copyright (C) 2026 0401chen。权利人已确认有权将本项目原创代码按 **GNU AGPL-3.0-only** 授权；完整英文文本见 [LICENSE](LICENSE)。第三方组件、代码和数据保留原许可证，不因项目声明而重新归属，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [third_party](third_party/README.md)。

PyMuPDF/MuPDF 选择上游 AGPL 许可。本系统集成 AGPL 组件；向外部用户提供网络交互时，应提供所部署版本完整对应源码（包括应用集成与修改），并满足 AGPL 第13条的源码获取要求。不可访问的 Private 仓库链接不能满足对外服务的源代码提供义务。发布二进制、容器或上线服务前，还需按[许可检查](docs/LICENSE_REVIEW.md)提供完整对应源码及依赖来源。
