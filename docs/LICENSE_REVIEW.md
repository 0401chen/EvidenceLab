# 公开候选许可检查

对象：本版本源码、配置模板、原创合成样例、适用测试及第三方许可证；不含旧 Git 历史或私人研究材料。

## 原创与来源

权利人已确认全部原创代码均有权按 AGPL-3.0-only 授权。审查未发现另一个项目的实现文件被复制后改署原创，底层能力通过依赖调用提供。新仓库只建立使用 GitHub noreply 邮箱的根提交，不继承旧历史私人邮箱。第三方通知的原始版权姓名/邮箱依法保留。

根 LICENSE 来自 GNU 官方 `https://www.gnu.org/licenses/agpl-3.0.txt`，完整未经修改，34,523 字节，SHA256：`0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`。README 和元数据选择 AGPL-3.0-only；许可证末尾示例不把本项目改为“or later”。

## 依赖检查

- PyMuPDF/MuPDF 选择 AGPL，与本项目 AGPLv3 源码发布方向一致；未声称取得商业授权。[官方说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)、[Artifex 条件](https://artifex.com/licensing)。
- MIT、BSD、ISC、Apache-2.0、PSF-2.0 组件保留原许可和版权；Apache 通知与专利条件仍有效。
- certifi 的 MPL-2.0、orjson 的 `MPL-2.0 AND (Apache-2.0 OR MIT)` 单独记录，全文保留；不改写为项目拥有的 AGPL 代码。组件未修改，源码来源保留。
- caniuse-lite 数据是 CC-BY-4.0，保留许可并注明 caniuse/caniuse-lite 来源，未修改数据。
- Python、Node.js/npm 由使用者安装，仓库未捆绑运行时、依赖二进制、模型权重或研究 PDF。

逐项版本与许可见[完整清单](../third_party/DEPENDENCIES.md)，原始文本见 `third_party/licenses/`。langsmith wheel 没有独立许可文件，额外从精确版本上游取得 MIT LICENSE。esbuild/Rollup 平台执行包采用同版本主包的原始 MIT 许可。无明确再分发依据的历史论文内容和截图未进入候选。

## 发布条件

当前候选为源码发布，依赖从软件源安装。未发现已知未解决的组件许可冲突；未来增加依赖或材料仍须重新检查。

Private 候选库不是公众可访问的源代码提供方式。切换 Public 后须核验未登录用户能读取对应源码与许可。本系统集成 AGPL 组件，对外提供网络交互时须向用户提供所部署版本的完整对应源码（含应用集成与修改），并满足 AGPL 第13条要求；前端已有源码和许可入口。分发二进制/容器须提供构建脚本、确切依赖和相应条件要求的依赖源码，不能只给不可访问的 Private 链接。本轮未制作二进制包、容器或上线公网服务。
