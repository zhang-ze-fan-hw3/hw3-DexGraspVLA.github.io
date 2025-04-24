<h1 align="center"> DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping </h1>


### 📝 [Paper](https://arxiv.org/abs/2502.20900) | 🌍 [Project Page](https://dexgraspvla.github.io/) | 📺 [Video](https://www.youtube.com/watch?v=X0Sq7q-bfI8)


![](./assets/teaser.jpg)


# 摘要

&emsp;&emsp;`DexGraspVLA`是一个**分层式视觉-语言-动作 (Vision-Language-Action) 框架**，旨在实现通用的灵巧抓取。其核心创新点在于：

1.  **强大的泛化能力**：在包含**数千种未见过的**物体、光照和背景组合的“**零样本**”真实世界环境中，实现了超过 **90%** 的**灵巧抓取**成功率，尤其是在**杂乱场景**中表现出色。
2.  **复杂的推理与长时序任务**：能够理解并执行需要**复杂视觉语言推理**的指令，完成**长时序抓取任务**。
3.  **分层架构**：利用预训练的**视觉语言大模型 (VLM)**（如 Qwen-VL）作为**高层任务规划器 (Planner)** 进行理解和决策，并学习一个基于**扩散模型 (Diffusion Model)** 的策略作为**底层动作控制器 (Controller)** 来生成灵巧的动作。
4.  **关键思路**：有效利用**基础模型 (Foundation Models)** 实现强大的泛化性，并通过**基于扩散的模仿学习 (Diffusion-based Imitation Learning)** 来学习灵巧的操作技能。
5.  **模块化设计**：整个框架采用模块化设计，便于训练、调试和扩展。

![](./assets/method.jpg)

# 环境配置

首先，请创建并激活`conda`环境：

```bash
conda create -n dexgraspvla python=3.9
conda activate dexgraspvla
git clone https://github.com/Psi-Robot/DexGraspVLA.git
cd DexGraspVLA
pip install -r requirements.txt
```

&emsp;&emsp;然后，请按照官方说明安装 [SAM](https://github.com/facebookresearch/segment-anything)和[Cutie](https://github.com/hkchengrex/Cutie)。
`DexGraspVLA`使用的`CUDA`版本是`12.6`。

# DexGraspVLA 推理 (Inference)

`DexGraspVLA`用于灵巧抓取的硬件平台如下图所示。

![](./assets/hardware.jpg)

&emsp;&emsp;由于知识产权限制，`DexGraspVLA`无法开源与硬件相关的代码。但是，`DexGraspVLA`发布了其余代码供参考，并在下面提供了如何在此平台上运行`DexGraspVLA`的说明。

## 安装

首先，安装所需的依赖项：

```bash
pip install pymodbus==2.5.3 pyrealsense2==2.55.1.6486
```

## 配置
1. 硬件设置:

在 `inference_utils/config.yaml` 中配置硬件设置。
2. 控制器检查点:

&emsp;&emsp;在 `controller/config/train_dexgraspvla_controller_workspace.yaml` 中指定训练好的控制器模型检查点。
或者，用户可以使用`DexGraspVLA`预训练的检查点进行快速部署：
[dexgraspvla-controller-20250320](https://drive.google.com/file/d/1ge1FYD2wUqBnFewWzpsjQ5v6pEDBraOH/view?usp=sharing)。

## 自定义推理命令

根据用户需求修改 `inference.sh`，调整以下参数：

- `--manual`: 启用手动模式，允许用户手动标记边界框、监控抓取过程并在必要时重置。如果省略此参数，则使用完整的`DexGraspVLA`规划器，利用视觉语言模型`(VLM)`自主规划和监控抓取轨迹。
- `--save_deployment_data`: 保存推理回合的`rollout`数据，包括原始数据和录制的视频。
- `--gen_attn_map`: 生成并保存控制器的注意力图。

## 运行推理
一切设置完成后，使用以下命令启动推理过程：
```bash
./inference.sh
```

此命令将在指定的硬件平台上执行配置好的抓取流程。

执行期间，详细的日志会生成并存储在`logs`目录中。这些日志包括：

- 流程状态 – 抓取过程的实时更新
- 相机图像 – 执行过程中捕获的帧
- 规划器提示与响应 – 视觉语言模型`(VLM)`的输入和输出
- 可选数据 – 注意力图和`rollout`数据（如果启用）

示例日志可以在[这里](https://drive.google.com/file/d/1s6axQUKc6itKfpsIP1zTNX4Khu0yBTn9/view?usp=sharing)下载。

# DexGraspVLA 控制器 (Controller)
## 准备数据集

&emsp;&emsp;`DexGraspVLA`提供了一个小型的[数据集](https://drive.google.com/file/d/1Z4QIibZwudz_qUazAGQAF7lAFAoRROnK/view?usp=drive_link)，包含`51`个人类示教数据样本，让用户能够了解`DexGraspVLA`数据的内​​容和格式，并运行代码以亲身体验训练过程。

首先，在仓库根目录下创建一个`data`文件夹：
```bash
[DexGraspVLA]$ mkdir data && cd data
```

下载数据集并将其放入`data`文件夹中。然后，解压缩数据集：
```bash
[data]$ tar -zxvf grasp_demo_example.tar.gz && rm -rf grasp_demo_example.tar.gz
```
解压后，您会发现数据集以 [Zarr 格式](https://zarr.readthedocs.io/en/stable/) 组织，包含以下组：

## 数据集结构
### `data`组
- **action**: (K, 13)
  - 包含每个时间步的右机械臂和手的动作数据，由 13 个自由度 (DoF) 表示。
- **right_state**: (K, 13)
  - 包含每个时间步的右机械臂和手的状态数据，由 13 个自由度表示。
- **rgbm**: (K, H, W, 4)
  - 来自头部相机的第三视角图像，有 4 个通道，前 3 个通道是 RGB，第 4 个通道是二值掩码。
- **right_cam_img**: (K, H, W, 3)
  - 来自腕部相机的第一视角图像，有 3 个 RGB 通道。

### `meta`组
- **episode_ends**: (J,)
  - 标记每个示教回合的结束索引，用于分割不同的示教序列。

这里，K 代表总样本数，J 代表示教回合数。

## 启动训练
要在单个`GPU`上训练`DexGraspVLA`控制器，请运行：
```bash
python train.py --config-name train_dexgraspvla_controller_workspace
```
&emsp;&emsp;要在`8`个`GPU`上训练`DexGraspVLA`控制器，首先使用`accelerate config`配置`accelerate`（`DexGraspVLA`启用了 BF16 混合精度训练），然后运行`./train.sh`或：
```bash
accelerate launch --num_processes=8 train.py --config-name train_dexgraspvla_controller_workspace
```
&emsp;&emsp;用户还可以通过在`controller/config/train_dexgraspvla_controller_workspace.yaml`中指定`policy.start_ckpt_path`来从现有检查点开始训练。为了支持应用和微调，`DexGraspVLA`提供了一个开源的、高性能的模型检查点([dexgraspvla-controller-20250320](https://drive.google.com/file/d/1ge1FYD2wUqBnFewWzpsjQ5v6pEDBraOH/view?usp=sharing))，该检查点在发布时已在五个零样本位置部署和评估，展示了强大的泛化能力。此外，其他训练设置也可以通过修改`controller/config`文件夹中的配置文件进行自定义。

&emsp;&emsp;为了帮助理解内部模型行为，`DexGraspVLA`提供了生成、保存和可视化控制器注意力图的功能。要启用此功能，请在训练前将配置文件中的`gen_attn_map`设置为`True`。在每个采样步骤中，注意力图将作为`pickle`文件保存在实验目录下的`train_sample_attn_maps`文件夹中。要可视化它们，请运行`python attention_map_visualizer.py --attn_maps_dir <path to train_sample_attn_maps>`。这将在`train_sample_attn_maps`内新创建的、与相应`pickle`文件同名的文件夹下生成注意力图的图像。

# DexGraspVLA 规划器 (Planner)

&emsp;&emsp;`DexGraspVLA`在`planner`目录中提供了基于[Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)的`DexGraspVLA`规划器代码。`DexGraspVLA`的接口目前支持调用`API`或查询部署在云服务器上的模型。
```python
# Instantiate a planner that calls the API
planner = DexGraspVLAPlanner(
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen2.5-vl-72b-instruct"
)

# Instantiate a planner that queries a deployed model
planner = DexGraspVLAPlanner(
    base_url="your_deployed_model_url"
)
```
&emsp;&emsp;对于部署，`DexGraspVLA`利用一个`8`卡`A800 GPU`服务器来托管`Qwen2.5-VL-72B-Instruct`模型。部署使用 `vllm`版本`0.7.3`进行管理，并利用`Qwen2.5-VL-7B-Instruct`模型进行推测解码 (speculative decoding)。部署过程使用四块`GPU`。

使用以下命令部署模型：

```bash
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8001 \
 --model <path to Qwen2.5-VL-72B-Instruct> --seed 42 -tp 1 \
 --speculative_model <path to Qwen2.5-VL-7B-Instruct> --num_speculative_tokens 5 \
 --gpu_memory_utilization 0.9 --tensor-parallel-size 4
```

# 引用 (Citation)
如果`DexGraspVLA`的项目对您有帮助，请考虑引用：

```bibtex
@misc{zhong2025dexgraspvla,
      title={DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping}, 
      author={Yifan Zhong and Xuchuan Huang and Ruochong Li and Ceyao Zhang and Yitao Liang and Yaodong Yang and Yuanpei Chen},
      year={2025},
      eprint={2502.20900},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2502.20900}, 
}
```

# 致谢 (Acknowledgements)
&emsp;&emsp;此代码库基于[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [DiT](https://github.com/facebookresearch/DiT)和 [pi_zero_pytorch](https://github.com/lucidrains/pi-zero-pytorch/)。