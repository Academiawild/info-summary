# Codex 工作规则

请全程使用中文。

我是 GitHub 新手，不会编程。
请用新手能听懂的话解释，不要使用太多技术术语。

这是我的 GitHub 账号：

[https://github.com/Academiawild/](https://github.com/Academiawild/)

这是一个长期使用的项目规则文件。
以后我会不断提出新需求，请你根据我的具体需求修改代码，同时保持项目标准、清楚、容易维护。

## 总目标

每次改代码时，请顺手保持项目：

* 结构清楚
* 文件摆放合理
* 入口明确
* 命名统一
* 没有多余入口
* 没有重复功能
* README 简单易懂
* 新手打开仓库后知道该看哪里、改哪里、怎么运行

重要：
不要每次都重新大整理仓库。
如果项目结构已经清楚，就只修改本次需求相关的地方。
只有发现明显混乱、重复、过时、多余入口时，才进行整理。

## 每次开始前

在修改前，请先简单判断：

* 当前需求要改哪些文件
* 主要入口文件是否清楚
* 是否有重复入口
* 是否有多余文件
* 是否有命名不统一
* 是否有配置路径需要同步修改
* README 是否需要跟着更新

不要只看文件名，要结合代码、配置和运行方式一起判断。

## 修改代码时

请根据我的具体需求完成代码修改。

同时注意：

* 不要制造新的重复入口
* 不要随便新增杂乱文件
* 新文件要放在合适的位置
* 文件名和文件夹名用简单、清楚、标准的英文
* 代码里的名称也要尽量统一
* 如果移动、删除、重命名文件，请同步更新所有引用路径
* 保证项目还能正常运行

如果发现旧文件、旧入口、测试残留、临时文件、重复功能已经没用了，可以整理或删除。
但如果可能删除核心功能、影响运行方式，或者改变项目主要逻辑，请先说明原因。

## README 要求

如果本次改动影响了运行方式、主要文件、配置文件或输出结果，请同步更新 README。

README 请保持中文，并写给 GitHub 新手看。

README 只保留真正需要看的内容：

* 这个工具是做什么的
* 我主要看哪个文件
* 我主要改哪个文件
* 需要改哪几行
* 怎么运行
* 运行后结果在哪里
* 哪些文件不要上传到 GitHub

请删除或避免：

* 冗长内容
* 过时内容
* 重复说明
* 多余入口说明
* 新手看不懂、也用不到的开发者说明

## 每次完成后

请用简单的话告诉我：

* 这次改了哪些文件
* 为什么要这样改
* 有没有整理冗余入口或多余文件
* 有没有更新 README
* 我以后主要看哪个文件
* 我以后主要改哪个文件
* 我该怎么运行
* 运行后结果在哪里
* 哪些文件不要上传到 GitHub

请不要只给结论，要告诉我为什么这样做。
解释要简单，适合不会编程的新手。

## 本项目作为工具使用

以下说明供 AI 助手使用；README 只保留用户日常需要的内容。

统一入口为 `python -m info_summary`：无参数打开窗口，带命令直接处理。先在项目目录运行 `.venv/bin/python -m info_summary --help` 查看帮助；首次缺少组件时运行 `./scripts/setup_app.sh`。无需操作窗口或另写临时调用脚本。

```bash
# 提取原文，也可以传音频文件或文件夹路径
.venv/bin/python -m info_summary process "视频链接"

# 原文加总结，保存为 Markdown
.venv/bin/python -m info_summary process "视频链接" --preset summary --format md

# 生成可离线搜索的图文阅读报告
.venv/bin/python -m info_summary process "视频链接" --preset summary --format html

# 结合画面：原文、画面笔记和综合总结
.venv/bin/python -m info_summary process "视频链接" --visual --preset summary

# 指定重点补充，另存新版，复用有效进度（不要用 --force）
.venv/bin/python -m info_summary process "视频链接" --visual-focus "把 05:20 附近的图表讲细一点" --preset summary

# 下载视频
.venv/bin/python -m info_summary download "视频链接" --quality 1080
```

- 命令行默认只生成 Word 原文稿，不调用文字模型；窗口新用户默认生成阅读报告，旧格式选择保留并新增报告。按用户要求选择 `--preset summary`（原文和总结）、`polish`（原文和润色）或 `all`（全部），或明确开启画面功能时才使用模型。格式用 `--format html|docx|md|both`，`both` 仍只表示 Word 和 Markdown。
- 画面功能仅支持视频链接：`--visual` 生成带时间、截图和解读的画面笔记；`--visual-focus "重点要求"` 自动开启并定位重点。沿用同一模型设置，但模型和服务必须支持图片。先看代表画面，再根据笔记/总结中的具体缺口补图；够用或无新增有效信息就停止，不设置固定截图数量或返工轮数。原文/润色稿不混入视觉推测；综合总结仍须选择 `--preset summary|all`。截图在结果内的 `images/`，Word 内嵌，Markdown 使用相对路径；`output_files` 还包含所选格式的 `visual.docx|md` 和 `image.*`。
- HTML 将本次选择的内容集中保存为 `阅读报告.html`，JSON 使用 `output_files["report.html"]`，不生成各步骤的独立 HTML。样式、脚本、截图全部内嵌，可以离线阅读、搜索、打印或单独复制；不为展示额外调用模型。原始截图仍保留在 `images/` 并参加完整结果检查。窗口“查看结果”优先打开报告，旧结果打开已有文档，下载任务打开视频；命令不会自动打开文件。
- 同时选择总结与画面时，报告正文是一篇“图文讲解”：观点、对应截图、结合讲述的解释放在一起，不再分成总结和画面笔记两个大块。现有视觉总结调用保留程序提供的 `[[frame:N]]` 引用；HTML只据当前画面列表内联配图，Word/Markdown转换成可读时间标签。原文不含画面推测；旧总结仅按明确时间关联，不靠关键词猜图，无引用时保留按时间的图文解释及收起的旧总结。普通文字提示词和缓存保持兼容。
- 图文总结先交代核心判断，再按主题合并说明依据、条件和时间；正文仅配解释主要判断所需的截图。逐帧复述、新闻逐条反驳、鼠标读数、绘图控制点等核对细节留在可展开依据中，不冲淡主线；影响结论的限定仍须简洁保留。不要为了用完已提取截图而拉长正文，也不固定主题数量、字数或配图数量。
- 多个来源分别传入，含空格的路径加引号；相对路径按调用者当前目录解释。保存位置沿用窗口设置，未改过则为桌面；`--output "路径"` 可临时指定。
- 模型和网站访问配置沿用已保存设置。`--provider opencode|ollama`、`--model`、`--base-url` 可临时覆盖模型；密钥用 `INFO_SUMMARY_API_KEY` 环境变量提供，不接受密钥命令参数，也不要打印设置文件或密钥。在线服务需要兼容 Chat Completions，基础地址通常以 `/v1` 结尾；“读取模型”失败不代表手动指定模型无法生成。
- 命令不弹窗、不要求确认，也不修改窗口偏好；画面开关和重点默认不继承窗口，需要显式指定。同一保存位置已有完整结果时自动复用，`--force` 会清除该来源续做进度并重新生成，指定重点返工通常不要用它。失败后再次运行相同命令可复用已保存的文字、截图及解读；完整画面视频也会复用。结果会记录到“最近结果”。
- 除 `--help` 外，stdout 只输出一个 JSON：`status` 为 `completed`、`partial`、`failed`、`invalid` 或 `cancelled`；`results` 包含每项的状态（复用为 `reused`）、`result_dir`、`output_files` 和 `warnings`，路径为绝对路径；`errors` 包含来源和失败原因。日志走 stderr。退出码 `0` 为全部成功、`1` 为失败或部分完成、`2` 为输入或配置错误、`130` 为取消。部分完成不能当成全部成功，要保留已有文件并报告警告。

维护代码时：统一入口在 `info_summary/__main__.py`；窗口协调在 `info_summary/app.py`，后台执行在 `info_summary/task_runner.py`；界面、表单、任务列表、设置对话框在 `info_summary/ui/`。命令在 `info_summary/cli.py`，共用参数判断在 `info_summary/requests.py`。阅读报告在 `info_summary/html_report.py` 和 `info_summary/report_assets/`，导出协调在 `info_summary/exporters.py`。润色/总结规则在 `info_summary/text_pipeline.py`，文字进度在 `info_summary/checkpoints.py`，通用安全存储在 `info_summary/storage.py`。画面判断在 `info_summary/video/visual_analyzer.py`，画面缓存在 `visual_cache.py`，下载与抽图在 `visual_media.py`；网站访问规则在 `client.py`，字幕解析在 `subtitles.py`。旧任务加载时不改写，点击“继续未完成”仍按当前界面选项构建请求。缓存提示词、哈希、目录与旧记录格式应保持兼容。现有检查用 `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` 运行。
