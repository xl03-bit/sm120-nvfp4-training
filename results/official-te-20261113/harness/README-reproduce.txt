官方 Transformer Engine NVFP4 / RTX 5060 Ti 16GB 对照交付

先打开 report.html。results/comparison-default.json 为默认主对照机器可读表。
project/ 是 e31a84233c0563a4c7b34305f5e41095b354b86f 的独立项目快照。
TransformerEngine/ 是 464df759d5a4ed549f2ce99b58fe653178a252f5 的官方源码快照。
未修改用户原项目、Windows research 环境或其升级前备份。

口径
1. 用户要求不再复测自己的实现；主基线均取 project/results 的 Git 归档。
2. 默认主对照：te_default.py；官方 NVFP4BlockScaling()，无性能参数搜索，
   无项目压缩权重、压缩 Adam 状态、转置重算或持久量化权重缓存。
   FP32 全参数和官方 FusedAdam FP32 moments；超参数与原项目一致。
   CUDA Graph 的 capturable=True 是保留原计时法所需。
3. 辅助对照：te_adapter.py + te_*.json / curve_te_*.json，使用项目压缩
   权重/优化器状态但官方 NVFP4 计算。不能把它当作纯官方默认栈。
4. PaddedLinear 只保留逻辑形状/共享权重、对齐后裁切以及 BF16 autocast
   边界。后者与原 LoRA 函数相同，解决 LayerNorm 输出 FP32 时 TE RHT
   类型错误；未改变 LayerNorm 本身。初次失败保存在 results/attempts。
5. 默认 Linear 只替换原线性层，不把模型改写为 TE 整体 TransformerLayer，
   以保持原注意力/激活/归一化。低秩分支用普通两次 F.linear 表达式。

独立环境（WSL Ubuntu-22.04）
文本/LoRA：/home/lixin/te-nvfp4-compare-20261113/venv
图像：/home/lixin/te-nvfp4-compare-20261113/image-venv
源码构建：/home/lixin/te-nvfp4-compare-20261113/TransformerEngine
完整版本：results/environment.json、environment-pip-freeze.txt；
图像版本另见 image-environment.json、image-environment-pip-freeze.txt。

从新目录复现，或为 --output 指定新的文件名，避免覆盖已交付证据。
下列命令在独立工作目录运行，前缀 bash run_env.sh 使用隔离文本环境：
  bash run_env.sh python measure.py --backend te_default --task 453m --output results/new-default-453m.json
  bash run_env.sh python measure.py --backend te_default --task fineweb --seed 2031 --output results/new-default-fineweb.json
  bash run_env.sh python measure.py --backend te_default --task fineweb --curve --output results/new-default-8192.json
  bash run_env.sh python lora_compare.py bench --output results/new-official-lora-bench.json
  bash run_env.sh python lora_compare.py fineweb --output results/new-official-lora-fineweb.json

诊断与容量（输出路径写在对应脚本中；再次运行前另存现有结果）：
  bash run_env.sh python check_default_graph.py
  bash run_env.sh python probe_official.py
  bash run_env.sh python probe_graph_rng.py
  bash run_env.sh python profile_default_workload.py
  bash run_env.sh python default_capacity.py 4b
  bash run_image_env.sh python default_capacity.py h3

FLUX 官方默认：
  bash run_image_env.sh python image_default.py bench --mode preserve --label official-default-40 --steps 40 --warmup 10 --validate
  bash run_image_env.sh python image_default.py train --mode bf16 --attention cudnn --label official-default-1000 --steps 1000
注意 CLI 中 preserve/bf16 用于选择原归档的源模型/数据分支；注入的投影计算
始终为官方 TE NVFP4。状态文件明确记录后端/配方。标签目录已存在时拒绝覆盖。
官方从相同 BFL 修订的 BF16 原始检查点量化，不导入项目 Comfy 预量化 payload。
数据、1024px、批次1、rank/alpha16、98投影、checkpointing、cuDNN SDPA、
FP32 fused AdamW、clip=1、学习率/噪声日程/固定验证均沿用仓库脚本。

安装/构建记录
build.sh、resume_build.sh 与 logs/build.log 保留初始完整官方核心构建。
install_image_env.sh 创建独立图像环境并匹配归档 PyTorch2.14.1+cu132、
diffusers0.40.0、transformers5.16.1。
build_image_torch_only.sh 调用官方 pytorch/setup.py build_ext --inplace，
仅重编译新 PyTorch 绑定，复用同提交未改动的官方核心；二进制哈希存档。
build_image_bindings.sh 是已停止的全量重复构建尝试，勿用它代替上述入口。
源码逻辑审计忽略 Windows CRLF 与权限位差异，原始状态另保留。

计时与显存
短测包含完整 forward/loss/backward/optimizer；graph 含输入复制。
长程8192与GPT2的评估点、数据窗口、优化器步数均存入JSON。
NVML 每20ms采样整卡驻留，包含非PyTorch显存及其他GPU用户；不是进程独占。
WDDM/WSL下分配器额度可大于物理VRAM，OOM报错的超大进程占用数字还有
无效计数现象。保留原日志，使用NVML描述实际整卡驻留，不能把30GiB的
分配额度说成16GB卡有30GiB物理显存。未启用自定义CPU卸载。
归档没有NVML对照，故不提供虚构的整卡差值。

原生计算证据
official_probe.json / official-kernel-evidence.json：RHT/SR开关、操作数与trace。
official-graph-rng.json：图重放SR随机量化确实变化，关闭SR时逐字节一致。
default453-full-update-kernel-evidence.json：真实453M完整更新三个GEMM方向各64次。
native-fp4-sass-evidence.json / sass-module_39.txt：实际加载cubin中的E2M1 OMMA指令。
Nsight计数器因权限不足不可用；未改变驱动/安全设置。CUPTI只捕获模块，不开计数器。
官方RHT当前主要用于wgrad列布局；冻结LoRA基座没有基座wgrad，不能把配方开关
等同于每个LoRA算子都运行RHT。低秩适配矩阵仍是BF16计算/FP32存储。

重新生成报告（不训练）：
  python verify_results.py
  python finalize_metadata.py
  python make_report.py
  python finalize_metadata.py
  python package_delivery.py
make_report_hybrid_legacy.py 仅为历史草稿来源，最终交付使用 report_default.py。
结果覆盖清单在 results/coverage.json；文件哈希在 results/sha256-manifest.json。

交付与最终检查点
NVFP4-comparison-delivery.zip 包含测试代码、报告、项目基线快照、日志、
JSON原始结果、剖析trace和匹配原生FP4内核的module_39.cubin。
完整官方源码、已安装环境、源模型和训练检查点保留在本机，不重复装入ZIP。
最终FLUX适配器及Adam状态：
  C:\Users\18066\Desktop\workspace\te-nvfp4-compare-20261113\image-default\results\official-default-1000\adapter-1000.pt
CPU核验确认98个适配层、23,887,872个FP32可训练参数、196份Adam状态，
全部状态位于第1000步且数值有限；见results/image-checkpoint-integrity.json。
输入模型/缓存和实际加载cubin的SHA256在results/input-and-cubin-sha256.json。
FLUX1000步对照统计同时排除原归档分段恢复的预热更新，共970个相同计时索引；
所有1000步仍参与训练及数据顺序核验。均值和中位数均保留，不按速度挑选样本。
