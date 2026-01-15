# Whisper

[[博客]](https://openai.com/blog/whisper)
[[论文]](https://arxiv.org/abs/2212.04356)
[[模型卡片]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab 示例]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper 是一个通用的语音识别模型。它在大量多样化的音频数据集上进行训练，同时也是一个多任务模型，可以执行多语言语音识别、语音翻译和语言识别。

## 方法 (Approach)

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

一个 Transformer 序列到序列（Seq2Seq）模型在各种语音处理任务上进行训练，包括多语言语音识别、语音翻译、口语语言识别和语音活动检测。这些任务共同表示为解码器要预测的一系列标记（tokens），从而允许单个模型替代传统语音处理管道的许多阶段。多任务训练格式使用一组特殊标记作为任务说明符或分类目标。

## 安装 (Setup)

我们使用 Python 3.9.9 和 [PyTorch](https://pytorch.org/) 1.10.1 来训练和测试我们的模型，但代码库预期与 Python 3.8-3.11 和最近的 PyTorch 版本兼容。代码库还依赖于一些 Python 包，尤其是 [OpenAI 的 tiktoken](https://github.com/openai/tiktoken)，用于其快速的分词器实现。您可以使用以下命令下载并安装（或更新到）最新版本的 Whisper：

    pip install -U openai-whisper

或者，可以使用以下命令拉取并安装该仓库的最新提交及其 Python 依赖项：

    pip install git+https://github.com/openai/whisper.git

要将软件包更新到此存储库的最新版本，请运行：

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

它还需要在您的系统上安装命令行工具 [`ffmpeg`](https://ffmpeg.org/)，大多数包管理器都可以提供：

```bash
# 在 Ubuntu 或 Debian 上
sudo apt update && sudo apt install ffmpeg

# 在 Arch Linux 上
sudo pacman -S ffmpeg

# 在 MacOS 上使用 Homebrew (https://brew.sh/)
brew install ffmpeg

# 在 Windows 上使用 Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# 在 Windows 上使用 Scoop (https://scoop.sh/)
scoop install ffmpeg
```

如果 [tiktoken](https://github.com/openai/tiktoken) 没有为您的平台提供预构建的 wheel，您可能还需要安装 [`rust`](http://rust-lang.org)。如果在执行上面的 `pip install` 命令时看到安装错误，请按照 [入门页面](https://www.rust-lang.org/learn/get-started) 安装 Rust 开发环境。此外，您可能需要配置 `PATH` 环境变量，例如 `export PATH="$HOME/.cargo/bin:$PATH"`。如果安装失败并出现 `No module named 'setuptools_rust'`，则需要安装 `setuptools_rust`，例如运行：

```bash
pip install setuptools-rust
```

## 可用模型和语言 (Available models and languages)

有六种模型大小，其中四种有仅英语版本，提供速度和准确性的权衡。
以下是可用模型的名称及其近似内存需求和相对于大型模型的推理速度。
下面的相对速度是在 A100 上转录英语语音测量的，实际速度可能会因多种因素（包括语言、说话速度和可用硬件）而有很大差异。

|  尺寸  | 参数量 | 仅英语模型 | 多语言模型 | 所需显存 | 相对速度 |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

用于仅英语应用的 `.en` 模型往往表现更好，尤其是对于 `tiny.en` 和 `base.en` 模型。我们观察到 `small.en` 和 `medium.en` 模型的差异变得不那么显著。
此外，`turbo` 模型是 `large-v3` 的优化版本，它提供了更快的转录速度，而准确性仅有微小的下降。

Whisper 的表现因语言而异。下图显示了使用 WER（词错误率）或 CER（字符错误率，以*斜体*显示）在 Common Voice 15 和 Fleurs 数据集上评估的 `large-v3` 和 `large-v2` 模型的语言性能细分。与其他模型和数据集对应的更多 WER/CER 指标可以在 [论文](https://arxiv.org/abs/2212.04356) 的附录 D.1、D.2 和 D.4 中找到，以及附录 D.3 中的 BLEU（双语评估替补）翻译分数。

![按语言细分的 WER](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

## 命令行使用 (Command-line usage)

以下命令将使用 `turbo` 模型转录音频文件中的语音：

```bash
whisper audio.flac audio.mp3 audio.wav --model turbo
```

默认设置（选择 `turbo` 模型）非常适合转录英语。但是，**`turbo` 模型未针对翻译任务进行训练**。如果您需要**将非英语语音翻译成英语**，请使用**多语言模型**（`tiny`, `base`, `small`, `medium`, `large`）之一，而不是 `turbo`。

例如，要转录包含非英语语音的音频文件，您可以指定语言：

```bash
whisper japanese.wav --language Japanese
```

要将语音**翻译**成英语，请使用：

```bash
whisper japanese.wav --model medium --language Japanese --task translate
```

> **注意：** 即使指定了 `--task translate`，`turbo` 模型也会返回原始语言。请使用 `medium` 或 `large` 获取最佳翻译结果。

运行以下命令以查看所有可用选项：

```bash
whisper --help
```

请参阅 [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) 查看所有可用语言的列表。

## Python 使用 (Python usage)

也可以在 Python 中执行转录：

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

在内部，`transcribe()` 方法读取整个文件并使用 30 秒的滑动窗口处理音频，在每个窗口上执行自回归序列到序列预测。

下面是 `whisper.detect_language()` 和 `whisper.decode()` 的使用示例，它们提供对模型的较低级别访问。

```python
import whisper

model = whisper.load_model("turbo")

# 加载音频并将其填充/修剪以适应 30 秒
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# 制作 log-Mel 频谱图并移至与模型相同的设备
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# 检测口语语言
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# 解码音频
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# 打印识别出的文本
print(result.text)
```

## 更多示例 (More examples)

请使用 Discussions 中的 [🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) 类别分享更多 Whisper 的示例用法以及第三方扩展，例如 Web 演示、与其他工具的集成、不同平台的移植等。

## 许可证 (License)

Whisper 的代码和模型权重在 MIT 许可证下发布。有关更多详细信息，请参阅 [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE)。
