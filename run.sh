#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   LIBERO + Isaac-GR00T Training Launcher        ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check if train.py exists
if [ ! -f "train.py" ]; then
    echo -e "${YELLOW}Error: train.py not found!${NC}"
    exit 1
fi

echo -e "\nSelect training mode:"
echo "1) Quick Test (10 steps, small batch)"
echo "2) Full Training (300 epochs)"
echo "3) Low Memory (Gradient accumulation)"
echo "4) Custom (Enter parameters)"
echo "5) Evaluate Model (MSE)"
echo "6) Visualize Predictions"
echo "q) Quit"

read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Starting Quick Test...${NC}"
        python3 train.py --preset quick_test
        ;;
    2)
        echo -e "\n${GREEN}Starting Full Training...${NC}"
        python3 train.py --preset full_training
        ;;
    3)
        echo -e "\n${GREEN}Starting Low Memory Training...${NC}"
        python3 train.py --preset low_memory
        ;;
    4)
        echo -e "\n${YELLOW}Enter custom parameters:${NC}"
        read -p "Output directory [output/custom]: " out_dir
        out_dir=${out_dir:-output/custom}
        
        read -p "Max steps [100]: " steps
        steps=${steps:-100}
        
        read -p "Batch size [4]: " batch
        batch=${batch:-4}
        
        echo -e "\n${GREEN}Starting Custom Training...${NC}"
        python3 train.py \
            --output_dir "$out_dir" \
            --max_steps "$steps" \
            --batch_size "$batch"
        ;;
    5)
        echo -e "\n${GREEN}Starting Evaluation...${NC}"
        read -p "Checkpoint path [output/libero_groot_training/checkpoint-20]: " ckpt
        ckpt=${ckpt:-output/libero_groot_training/checkpoint-20}
        python3 evaluate.py --checkpoint "$ckpt"
        ;;
    6)
        echo -e "\n${GREEN}Starting Visualization...${NC}"
        read -p "Checkpoint path [output/libero_groot_training/checkpoint-20]: " ckpt
        ckpt=${ckpt:-output/libero_groot_training/checkpoint-20}
        python3 visualize.py --checkpoint "$ckpt" --num-samples 5
        ;;
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
