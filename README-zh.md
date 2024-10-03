<div align="center">

# Invoke TIPO

_✨从输入的提示词生成详细的提示词_
![preview](./assets/image_1.png)
📓 · [Documents](./README.md) · [中文文档](./README-zh.md)
</div>


## 简介
一个为 [InvokeAI](https://github.com/invoke-ai/InvokeAI) 添加 TIPO 节点的扩展，可将输入的简单的提示词扩展为详细的提示词，提高图片生成的质量。该节点移植自 [z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension)。


## 安装
进入 InvokeAI 的节点目录（`invokeai/nodes`），若不清楚该路径在哪，可通过启动 InvokeAI 时终端显示的信息找到。

例如，InvokeAI 在启动时将显示 InvokeAI 的根目录。

```
[2024-10-03 22:01:25,401]::[InvokeAI]::INFO --> Root directory = E:\Softwares\InvokeAI\invokeai
```

从终端中可以知道 InvokeAI 的根目录在`E:\Softwares\InvokeAI\invokeai`，安装节点前就需要进入该目录中（`E:\Softwares\InvokeAI\invokeai\nodes`）。

进入 InvokeAI 的节点目录后，打开终端，输入下面的命令进行安装。

```
git clone https://github.com/licyk/invoke_tipo
```

或者将该 Github 仓库下载下来，并解压到该目录中。

安装完成后需重启 InvokeAI。


## 使用
进入 InvokeAI 的工作流中，在添加节点处搜索`TIPO`节点并添加。

在`invoke_tipo/workflow`中可有示例工作流，可导入并使用。


## 鸣谢
- [@KohakuBlueleaf](https://github.com/KohakuBlueleaf) - 提供 TIPO