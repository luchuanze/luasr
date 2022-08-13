<div align=center><img src ="https://user-images.githubusercontent.com/32317033/184311331-b98dbe19-e7e6-4b1d-bfdf-5fc809d7fcb6.png"/></div>

# LuASR is an end-to-end ASR project
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## 概述

#### LuASR 是一个端到端语音识别项目，目的是基于 PYTORCH 框架提供当前流行的 CTC , TRANSDUCE , TRANSFORMER 等多任务端到端识别架构，支持不同编码模块如，TDNN, LSTM, MHA, COMFORMER 供从事语音识别者学习；也将提供基于 C/C++的runtime （x86、ARM）解码器可用于项目工程化。

#### 该项目参考当前一些流行的语音识别开源项目，如 wenet，next gen kaldi, espent 等。

## 安装及使用

### 环境配置

使用 Linux 系统，推荐 Ubantu 20.+ 。

安装 Python 开发环境，推荐版本 >= 3.8 。

安装 pytorch 框架，推荐版本 >= 1.10  。

``` sh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
``` 

获取源码

``` sh
git clone https://github.com/luchuanze/luasr.git
```

### 模型训练
