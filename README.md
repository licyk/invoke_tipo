<div align="center"

# Invoke TIPO

_✨Generate detailed prompt from input tags_
![preview](./assets/image_1.png)
📓 · [Documents](./README.md) · [中文文档](./README-zh.md)
</div>


## Info  
An extension that adds a TIPO node for [InvokeAI](https://github.com/invoke-ai/InvokeAI), which can expand simple input prompts into detailed prompts to improve the quality of image generation. This node is ported from [z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension).


## Install
Navigate to the InvokeAI nodes directory (`invokeai/nodes`). If you are unsure of the path, you can find it by looking at the information displayed in the terminal when starting InvokeAI.

For example, when InvokeAI starts, it will display the root directory of InvokeAI.

```
[2024-10-03 22:01:25,401]::[InvokeAI]::INFO --> Root directory = E:\Softwares\InvokeAI\invokeai
```

From the terminal, you can see that the root directory of InvokeAI is at `E:\Softwares\InvokeAI\invokeai`, and you need to enter this directory (`E:\Softwares\InvokeAI\invokeai\nodes`) before installing the node.

Once you are in the InvokeAI nodes directory, open the terminal and enter the following command to install.

```
git clone https://github.com/licyk/invoke_tipo
```
 
Alternatively, download the GitHub repository and unzip it into the directory.

Restart InvokeAI after the installation is complete.


## Use
Within the InvokeAI workflow, search for the TIPO node and add it when you are at the point of adding nodes.

There are example workflows available in `invoke_tipo/workflow` that you can import and use.


## Acknowledgement
- [@KohakuBlueleaf](https://github.com/KohakuBlueleaf) - Provide TIPO.
