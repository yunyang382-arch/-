# Whisper 音视频转录与复盘辅助工具

这是一个基于 [OpenAI Whisper](https://github.com/openai/whisper) 构建的个人效率工具库，主要用于将股市复盘、课程讲解、会议记录等音视频内容自动转录为文字稿和 SRT 字幕，从而极大提高内容整理和知识内化的效率。

## 核心功能

*   **全自动批量处理**：通过 `reproduce_project.py` 脚本，自动扫描指定目录下的所有媒体文件，无需逐个手动操作。
*   **广泛的格式支持**：支持 `.mp3`, `.wav`, `.flac`, `.m4a`, `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm` 等主流音视频格式。
*   **生成的产物**：
    *   **TXT 纯文本**：便于快速浏览全文，提取核心观点（如“龙哥的操作规律”、“市场情绪分析”等）。
    *   **SRT 字幕文件**：包含精确的时间轴，方便在回看视频或重听音频时进行对照，精确定位关键语录。
*   **特定环境优化**：针对特定的 Windows Conda 环境进行了 ffmpeg 路径的自动适配。

## 快速开始

### 1. 环境准备

确保你的计算机上安装了 Python 以及必要的依赖库。

```bash
# 安装 OpenAI Whisper 及其依赖
pip install -U openai-whisper tqdm
```

此外，你需要安装 `ffmpeg` 并确保其在系统 PATH 中，或者在脚本中正确配置了其路径。

### 2. 配置路径

打开 `reproduce_project.py` 文件，找到 `recordings_dir` 变量，将其修改为你存放录音/视频文件的文件夹路径。

```python
# 例如：
recordings_dir = r"C:\Users\your_name\Downloads\recordings"
```

### 3. 运行转录

在终端中运行脚本：

```bash
python reproduce_project.py
```

脚本将开始：
1.  自动检测 ffmpeg 路径。
2.  加载 Whisper 模型（默认为 `tiny` 模型，可根据显存大小修改为 `small`, `base`, `medium` 或 `large`）。
3.  遍历目录下未处理的文件并进行转录。
4.  实时打印转录进度。

### 4. 输出结果

转录完成后，在你的源文件目录下会自动生成同名的 `.txt` 和 `.srt` 文件。

## 工作流建议

1.  **录制/下载**：获取你需要复盘的音频或视频（如盘中解盘、技术分析录屏）。
2.  **转录**：使用本工具一键生成文字。
3.  **整理**：使用 Markdown 软件（如 Obsidian, Typora）打开生成的 TXT，结合 SRT 的时间戳，提炼出关键策略和逻辑（如“龙头战法”、“情绪周期”等）。
4.  **归档**：将整理好的 `.md` 笔记保存在本项目中进行版本管理。

## 原始项目

本项目核心依赖于 [OpenAI Whisper](https://github.com/openai/whisper)。
Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.
