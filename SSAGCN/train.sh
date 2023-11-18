# !/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_ssagcn 1 --n_txpcnn 7  --dataset eth --tag ssagcn-eth --use_lrschd --num_epochs 500 && echo "eth Launched."&
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_ssagcn 1 --n_txpcnn 7  --dataset hotel --tag ssagcn-hotel --use_lrschd --num_epochs 500 && echo "hotel Launched." &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_ssagcn 1 --n_txpcnn 7  --dataset univ --tag ssagcn-univ --use_lrschd --num_epochs 500 && echo "univ Launched." &
P2=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_ssagcn 1 --n_txpcnn 7  --dataset zara1 --tag ssagcn-zara1 --use_lrschd --num_epochs 500 && echo "zara1 Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_ssagcn 1 --n_txpcnn 7  --dataset zara2 --tag ssagcn-zara2 --use_lrschd --num_epochs 500 && echo "zara2 Launched." &
P4=$!

wait $P0 $P1 $P2 $P3 $P4

