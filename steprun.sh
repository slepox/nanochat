#!/bin/bash

# 分步骤执行的nanochat训练脚本
# 使用方法: bash step_run.sh [step1] [step2] ...
# 可用步骤: setup, tokenizer, base_model, mid_training, sft, rl, report
# 示例: bash step_run.sh setup tokenizer
# 如果不指定步骤，将显示帮助信息

# 默认中间产物目录
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 数据集存储目录
export DATA_DIR="$NANOCHAT_BASE_DIR/base_data"

# wandb设置
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# 检测系统和硬件
 detect_system_and_hardware() {
    # 检测操作系统
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS系统
        echo "检测到macOS系统"
        
        # 检测Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo "检测到Apple Silicon (M1/M2/M3)"
            DEVICE_TYPE="mps"
            NPROC_PER_NODE=1  # Apple Silicon目前不支持多GPU分布式训练
        else
            echo "检测到Intel Mac"
            DEVICE_TYPE="cpu"
            NPROC_PER_NODE=1  # Intel Mac使用CPU
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux系统
        echo "检测到Linux系统"
        
        # 检测NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            echo "检测到 $GPU_COUNT 个NVIDIA GPU"
            DEVICE_TYPE="cuda"
            NPROC_PER_NODE=$GPU_COUNT
        else
            echo "未检测到NVIDIA GPU，使用CPU"
            DEVICE_TYPE="cpu"
            NPROC_PER_NODE=1
        fi
    else
        # 其他系统，默认使用CPU
        echo "检测到未知系统，默认使用CPU"
        DEVICE_TYPE="cpu"
        NPROC_PER_NODE=1
    fi
    
    echo "设备类型: $DEVICE_TYPE, 进程数: $NPROC_PER_NODE"
}

# 精度设置
export NANOCHAT_PRECISION="bf16"

# 数据集下载进程ID
DATASET_DOWNLOAD_PID=""

# 检查数据集是否已下载
dataset_is_downloaded() {
    local shards=$1

    # 确保数据目录存在
    if [ ! -d "$DATA_DIR" ]; then
        return 1
    fi

    # 检查前N个分片文件是否都存在
    # 文件名格式: shard_00000.parquet, shard_00001.parquet, ...
    local missing=0
    for ((i=0; i<shards; i++)); do
        local filename=$(printf "shard_%05d.parquet" $i)
        local filepath="$DATA_DIR/$filename"
        if [ ! -f "$filepath" ]; then
            missing=$((missing + 1))
        fi
    done

    # 如果所有文件都存在，返回成功
    if [ $missing -eq 0 ]; then
        return 0
    else
        echo "缺少 $missing 个分片文件（总共需要 $shards 个）"
        return 1
    fi
}

# 获取已下载的数据集分片数量
get_downloaded_shards_count() {
    if [ ! -d "$DATA_DIR" ]; then
        echo 0
        return
    fi

    # 统计 shard_*.parquet 文件数量（排除临时文件）
    local count=$(find "$DATA_DIR" -name "shard_*.parquet" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
    echo $count
}

# 设置步骤
setup() {
    echo "===== 开始设置环境 ====="

    # 检查并安装 Rust/Cargo
    echo "检查并安装 Rust..."
    if ! command -v rustc &> /dev/null || ! command -v cargo &> /dev/null; then
        echo "Rust 未安装，开始安装..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    else
        echo "Rust 已安装，跳过安装"
        # 确保环境变量已加载
        source "$HOME/.cargo/env"
    fi

    # install uv (if not already installed)
    echo "检查并安装 uv..."
    if ! command -v uv &> /dev/null; then
        echo "uv 未安装，开始安装..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        echo "uv 已安装，跳过安装"
    fi
    
    # create a .venv local virtual environment (if it doesn't exist)
    echo "检查并创建虚拟环境..."
    [ -d ".venv" ] || uv venv
    
    # install the repo dependencies
    echo "安装项目依赖..."
    
    # 检测系统和硬件
    detect_system_and_hardware
    
    # 根据系统选择合适的依赖
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        echo "安装GPU版本依赖..."
        uv sync --extra gpu
    else
        echo "安装CPU版本依赖（包含Apple Silicon MPS支持）..."
        uv sync --extra cpu
    fi
    
    # activate venv
    source .venv/bin/activate
    
    echo "===== 环境设置完成 ====="
}

# 数据集下载步骤
download_dataset() {
    echo "===== 开始下载数据集 ====="
    # 确保环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "请先执行 setup 步骤"
        return 1
    fi
    
    local shards=240
    
    # 解析参数
    while [ $# -gt 0 ]; do
        case "$1" in
            --shards=*)
                shards="${1#--shards=}"
                shift
                ;;
            *)
                echo "未知参数: $1"
                echo "用法: download_dataset [--shards=N]"
                return 1
                ;;
        esac
    done
    
    # 检查已下载的分片数量
    local current_count=$(get_downloaded_shards_count)
    echo "当前已下载 $current_count 个数据分片"

    # 下载指定数量的数据分片
    if dataset_is_downloaded $shards; then
        echo "$shards 个数据分片已全部存在，跳过下载"
    else
        echo "下载 $shards 个数据分片..."
        python -m nanochat.dataset -n $shards
        if [ $? -eq 0 ]; then
            local new_count=$(get_downloaded_shards_count)
            echo "$shards 个数据分片下载完成（当前共有 $new_count 个分片）"
        else
            echo "数据集下载失败"
            return 1
        fi
    fi
    
    echo "===== 数据集下载步骤完成 ====="
}

# 分词器步骤
tokenizer() {
    echo "===== 开始设置和训练分词器 ====="
    # 确保环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "请先执行 setup 步骤"
        return 1
    fi
    
    # Rust 环境已在 setup 步骤中配置
    
    # Build the rustbpe Tokenizer
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
    
    # 重置报告
    python -m nanochat.report reset
    
    # 检查并下载初始数据集（如果需要）
    local current_count=$(get_downloaded_shards_count)
    echo "当前已下载 $current_count 个数据分片"

    if ! dataset_is_downloaded 8; then
        echo "初始8个数据分片尚未完全下载，开始下载..."
        python -m nanochat.dataset -n 8
        if [ $? -eq 0 ]; then
            local new_count=$(get_downloaded_shards_count)
            echo "初始8个数据分片下载完成（当前共有 $new_count 个分片）"
        else
            echo "初始数据集下载失败"
            return 1
        fi
    else
        echo "初始8个数据分片已存在，跳过下载"
    fi
    
    # 训练分词器
    python -m scripts.tok_train --max_chars=2000000000
    python -m scripts.tok_eval
    echo "===== 分词器设置和训练完成 ====="
}

# 基础模型步骤
base_model() {
    echo "===== 开始训练基础模型 ====="
    # 确保环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "请先执行 setup 步骤"
        return 1
    fi
    
    # 检查分词器是否已训练
    if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer.model" ]; then
        echo "错误: 分词器尚未训练，请先执行 tokenizer 步骤"
        return 1
    fi
    
    # 检查数据集是否已完全下载（需要240个分片）
    if ! dataset_is_downloaded 240; then
        echo "错误: 数据集尚未完全下载（需要240个分片），请先执行 download_dataset 步骤"
        return 1
    fi
    
    # 预训练模型
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
    else
        # Mac系统使用单进程
        python -m scripts.base_train -- --depth=20 --run=$WANDB_RUN --device_type=$DEVICE_TYPE
    fi
    # 评估模型
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
    else
        # Mac系统使用单进程
        python -m scripts.base_loss -- --device_type=$DEVICE_TYPE
        python -m scripts.base_eval -- --device_type=$DEVICE_TYPE
    fi
    echo "===== 基础模型训练完成 ====="
}

# 中间训练步骤
mid_training() {
    echo "===== 开始中间训练 ====="
    # 确保环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "请先执行 setup 步骤"
        return 1
    fi
    
    # 检查基础模型是否已训练
    if [ ! -d "$NANOCHAT_BASE_DIR/checkpoints/base" ]; then
        echo "错误: 基础模型尚未训练，请先执行 base_model 步骤"
        return 1
    fi
    
    # 检查数据集是否已完全下载
    if ! dataset_is_downloaded 240; then
        echo "错误: 数据集尚未完全下载（需要240个分片），请先执行 download_dataset 步骤"
        return 1
    fi
    
    # 下载身份对话数据
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    
    # 运行中间训练和评估
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
    else
        # Mac系统使用单进程
        python -m scripts.mid_train -- --run=$WANDB_RUN --device_type=$DEVICE_TYPE
        python -m scripts.chat_eval -- -i mid --device_type=$DEVICE_TYPE
    fi
    echo "===== 中间训练完成 ====="
}

# 监督微调步骤
sft() {
    echo "===== 开始监督微调 ====="
    # 确保环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "请先执行 setup 步骤"
        return 1
    fi
    
    # 检查中间训练是否已完成
    if [ ! -d "$NANOCHAT_BASE_DIR/checkpoints/mid" ]; then
        echo "错误: 中间训练尚未完成，请先执行 mid_training 步骤"
        return 1
    fi
    
    # 检查数据集是否已完全下载
    if ! dataset_is_downloaded 240; then
        echo "错误: 数据集尚未完全下载（需要240个分片），请先执行 download_dataset 步骤"
        return 1
    fi
    
    # 训练和评估
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
    else
        # Mac系统使用单进程
        python -m scripts.chat_sft -- --run=$WANDB_RUN --device_type=$DEVICE_TYPE
        python -m scripts.chat_eval -- -i sft --device_type=$DEVICE_TYPE
    fi
    echo "===== 监督微调完成 ====="
    echo "您现在可以使用以下命令与模型交互:"
    echo "  python -m scripts.chat_cli -p \"Why is the sky blue?\""
    echo "  python -m scripts.chat_web"
}

# 强化学习步骤（可选）
rl() {
    echo "===== 开始强化学习 ====="
    # 确保环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "请先执行 setup 步骤"
        return 1
    fi
    
    # 检查监督微调是否已完成
    if [ ! -d "$NANOCHAT_BASE_DIR/checkpoints/sft" ]; then
        echo "错误: 监督微调尚未完成，请先执行 sft 步骤"
        return 1
    fi
    
    # 检查数据集是否已完全下载
    if ! dataset_is_downloaded 240; then
        echo "错误: 数据集尚未完全下载（需要240个分片），请先执行 download_dataset 步骤"
        return 1
    fi
    
    # 运行强化学习
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K
    else
        # Mac系统使用单进程
        python -m scripts.chat_rl -- --run=$WANDB_RUN --device_type=$DEVICE_TYPE
        python -m scripts.chat_eval -- -i rl -a GSM8K --device_type=$DEVICE_TYPE
    fi
    echo "===== 强化学习完成 ====="
}

# 生成报告步骤
report() {
    echo "===== 生成完整报告 ====="
    # 确保环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "请先执行 setup 步骤"
        return 1
    fi
    
    # 检查是否有任何训练结果可以生成报告
    if [ ! -d "$NANOCHAT_BASE_DIR/checkpoints" ] || [ -z "$(ls -A $NANOCHAT_BASE_DIR/checkpoints 2>/dev/null)" ]; then
        echo "警告: 未找到任何训练结果，报告可能不完整"
        echo "请至少执行 tokenizer 和 base_model 步骤"
    fi
    
    # 生成报告
    python -m nanochat.report generate
    echo "===== 报告生成完成 ====="
    echo "报告已保存到当前目录的 report.md"
}

# 显示帮助信息
show_help() {
    echo "用法: bash step_run.sh [step1] [step2] ..."
    echo ""
    echo "系统检测信息:"
    detect_system_and_hardware
    echo ""
    echo "可用步骤:"
    echo "  setup         - 设置Python和Rust环境"
    echo "  download_dataset [--shards=N] - 下载数据集，默认下载240个分片"
    echo "  tokenizer     - 设置和训练分词器（会自动下载8个初始分片）"
    echo "  base_model    - 训练基础模型（要求240个分片已下载）"
    echo "  mid_training  - 中间训练（要求240个分片已下载）"
    echo "  sft           - 监督微调（要求240个分片已下载）"
    echo "  rl            - 强化学习（可选，要求240个分片已下载）"
    echo "  report        - 生成完整报告"
    echo "  all           - 执行所有步骤（会并行下载240分片和训练分词器）"
    echo ""
    echo "Mac系统支持:"
    echo "  - Apple Silicon (M1/M2/M3): 自动使用MPS加速"
    echo "  - Intel Mac: 使用CPU模式"
    echo "  - 不支持多GPU分布式训练，使用单进程模式"
    echo ""
    echo "参数说明:"
    echo "  --shards=N    - 指定下载的数据集分片数量（仅适用于download_dataset步骤）"
    echo ""
    echo "示例:"
    echo "  bash step_run.sh setup download_dataset --shards=8"
    echo "  bash step_run.sh download_dataset"
    echo "  bash step_run.sh tokenizer"
    echo "  bash step_run.sh base_model mid_training sft"
}

# 执行所有步骤
run_all() {
    setup
    
    # 下载初始8个分片
    download_dataset --shards=8
    
    # 异步启动240分片下载
    if ! dataset_is_downloaded 240; then
        echo "异步启动240个数据分片下载..."
        (download_dataset --shards=240) &
        DATASET_DOWNLOAD_PID=$!
        echo "数据集下载进程ID: $DATASET_DOWNLOAD_PID"
    fi
    
    # 执行分词训练（可以与240分片下载并行）
    tokenizer
    
    # 等待240分片下载完成
    if [ ! -z "$DATASET_DOWNLOAD_PID" ]; then
        echo "等待240个数据分片下载完成..."
        wait $DATASET_DOWNLOAD_PID
        DATASET_DOWNLOAD_PID=""
    fi
    
    # 执行后续训练步骤
    base_model
    mid_training
    sft
    # 可选步骤
    # rl
    report
}

# 执行主函数
main() {
    # 检测系统和硬件（全局变量）
    detect_system_and_hardware
    
    # 如果没有参数，显示帮助
    if [ $# -eq 0 ]; then
        show_help
        return 0
    fi
    
    # 执行指定的步骤
    local i=1
    while [ $i -le $# ]; do
        local step=${!i}
        case "$step" in
            setup)
                setup
                i=$((i+1))
                ;;
            download_dataset)
                # 提取download_dataset的参数
                local args=()
                local j=$((i+1))
                while [ $j -le $# ] && [[ ${!j} == --* ]]; do
                    args+=($(echo "${!j}" | grep -v "^--async$"))
                    j=$((j+1))
                done
                download_dataset "${args[@]}"
                i=$j
                ;;
            tokenizer)
                tokenizer
                i=$((i+1))
                ;;
            base_model)
                base_model
                i=$((i+1))
                ;;
            mid_training)
                mid_training
                i=$((i+1))
                ;;
            sft)
                sft
                i=$((i+1))
                ;;
            rl)
                rl
                i=$((i+1))
                ;;
            report)
                report
                i=$((i+1))
                ;;
            all)
                run_all
                return 0
                ;;
            help | --help | -h)
                show_help
                return 0
                ;;
            *)
                echo "未知步骤: $step"
                show_help
                return 1
                ;;
        esac
    done
}

# 执行主函数
main "$@"