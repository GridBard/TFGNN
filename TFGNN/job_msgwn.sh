python train.py --fusion skip --epochs 100 --gpuid 3 --batch_size 64 --data data/shenzhen --save ./garage/shenzhen --num_nodes 156 --in_dim 1 --seq_length 4 --train_ratio 0.7 --val_ratio 0.1 --seq_len 4 --pre_len 4 --print_every 10 --week_len 3
python train.py --fusion start --epochs 100 --gpuid 4 --batch_size 64 --data data/shenzhen --save ./garage/shenzhen --num_nodes 156 --in_dim 1 --seq_length 4 --train_ratio 0.7 --val_ratio 0.1 --seq_len 4 --pre_len 4 --print_every 10 --week_len 2
python train.py --fusion hour --epochs 100 --gpuid 0 --batch_size 64 --data data/shenzhen --save ./garage/shenzhen --num_nodes 156 --in_dim 1 --seq_length 4 --train_ratio 0.7 --val_ratio 0.1 --seq_len 4 --pre_len 4 --print_every 10 --week_len 2
python train.py --fusion day --epochs 100 --gpuid 1 --batch_size 64 --data data/shenzhen --save ./garage/shenzhen --num_nodes 156 --in_dim 1 --seq_length 4 --train_ratio 0.7 --val_ratio 0.1 --seq_len 4 --pre_len 4 --print_every 10 --week_len 2
python train.py --fusion week --epochs 100 --gpuid 2 --batch_size 64 --data data/shenzhen --save ./garage/shenzhen --num_nodes 156 --in_dim 1 --seq_length 4 --train_ratio 0.7 --val_ratio 0.1 --seq_len 4 --pre_len 4 --print_every 10 --week_len 2

python train.py --fusion skip --epochs 100 --gpuid 5 --batch_size 64 --data data/shenzhen --save ./garage/shenzhen --num_nodes 156 --in_dim 1 --seq_length 4 --train_ratio 0.7 --val_ratio 0.1 --seq_len 4 --pre_len 4 --print_every 10 --week_len 2







