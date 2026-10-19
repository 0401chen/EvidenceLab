# 第三方组件声明

EvidenceLab 原创应用代码采用 AGPL-3.0-only。以下组件由各自权利人拥有，项目许可证不覆盖或替代其许可证。本仓库不捆绑依赖二进制、模型权重、研究 PDF 或第三方论文截图。

| 直接组件 | 实际用途 | 上游许可及来源 |
|---|---|---|
| PyMuPDF / MuPDF | PDF 文本、坐标、分页渲染 | AGPLv3 或商业双许可，本项目选择 AGPL；[官方说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)、[源码](https://github.com/pymupdf/PyMuPDF) |
| LangGraph / LangChain Core | 状态图编排及运行依赖 | MIT；[LangGraph](https://github.com/langchain-ai/langgraph)、[LangChain](https://github.com/langchain-ai/langchain) |
| FastAPI | HTTP API | MIT；[源码](https://github.com/fastapi/fastapi) |
| Uvicorn | ASGI 服务 | BSD-3-Clause；[源码](https://github.com/encode/uvicorn) |
| python-multipart | 上传解析 | Apache-2.0；[源码](https://github.com/Kludex/python-multipart) |
| HTTPX | 模型 HTTP 客户端 | BSD-3-Clause；[源码](https://github.com/encode/httpx) |
| python-dotenv | 本机配置 | BSD-3-Clause；[源码](https://github.com/theskumar/python-dotenv) |
| React / React DOM | 用户界面 | MIT；[源码](https://github.com/facebook/react) |
| Phosphor Icons | 界面图标 | MIT；[源码](https://github.com/phosphor-icons/react) |
| TypeScript | 类型检查（开发） | Apache-2.0；[源码](https://github.com/microsoft/TypeScript) |
| Vite / React 插件 | 构建（开发） | MIT；[源码](https://github.com/vitejs/vite) |
| pytest | 后端测试（开发） | MIT；[源码](https://github.com/pytest-dev/pytest) |

完整 Python 版本闭包和 npm 锁定依赖逐项列于[依赖许可清单](third_party/DEPENDENCIES.md)。可获取的原始 LICENSE/COPYING/NOTICE 未经修改保存在 [third_party/licenses](third_party/licenses/)。上游版权姓名、版权邮箱及免责声明属于依法保留的第三方声明，不表示这些作者参与 EvidenceLab 开发。

特别注意：certifi 使用 MPL-2.0；orjson 的实际表达式为 `MPL-2.0 AND (Apache-2.0 OR MIT)`，不能简写为 MIT。相关许可全文均保留。构建依赖 caniuse-lite 的兼容性数据采用 CC-BY-4.0，源于 [caniuse](https://github.com/Fyrd/caniuse) 和 [caniuse-lite](https://github.com/browserslist/caniuse-lite)，作者/贡献者、源码和许可保留，数据未修改。esbuild/Rollup 平台包适用对应版本上游 MIT 许可。

本项目没有将第三方实现改署为原创。历史可选集成已从此独立版本移除，不再作为组件、依赖或宣传参考。合成示例和测试科研段落由本项目编写，不对应真实论文或真实实验。

分发依赖二进制或提供修改后的网络服务时，仍须遵守完整源码提供、署名及通知保留条件，见[许可检查](docs/LICENSE_REVIEW.md)。
