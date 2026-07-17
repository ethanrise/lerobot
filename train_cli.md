
# act swanlab_local

## train:

```
HF_HUB_OFFLINE=1 lerobot-train \
  --dataset.repo_id=local/uqi_teleop_data_0514 \
  --dataset.root=datasets/uqi_teleop_data_0514_lerobot \
  --dataset.video_backend=torchcodec \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/uqi_act_bs8_20k_seed1000_v1 \
  --job_name=uqi_act_bs8_20k_seed1000_v1 \
  --policy.device=cuda \
  --steps=20000 \
  --batch_size=8 \
  --num_workers=4 \
  --env_eval_freq=0 \
  --save_freq=5000 \
  --wandb.enable=false \
  --swanlab.enable=true \
  --swanlab.mode=local \
  --swanlab.project=uqi_teleop_0514
```

## view loss:

```
swanlab watch outputs/train/uqi_act_bs8_20k_seed1000_v1/swanlog --host 0.0.0.0 --port 5092
```

## View train result

```
HF_HUB_OFFLINE=1 python examples/port_datasets/check_act_policy_one_frame.py \
  --dataset.root=datasets/uqi_teleop_data_0514_lerobot \
  --checkpoint=outputs/train/uqi_act_bs8_20k_seed1000_v1/checkpoints/020000/pretrained_model
```

## compare all train results

```
HF_HUB_OFFLINE=1 python examples/port_datasets/eval_act_checkpoints_offline.py \
  --model-name=act \
  --run-name=uqi_act_bs8_20k_seed1000_v1 \
  --dataset.root=datasets/uqi_teleop_data_0514_lerobot \
  --checkpoint-glob='outputs/train/uqi_act_bs8_20k_seed1000_v1/checkpoints/*/pretrained_model' \
  --num-frames=1000 \
  --stride=10 \
  --output-dir=outputs/eval/act_uqi_act_bs8_20k_seed1000_v1_ckpt_compare \
  --plot-all
```

# diffusion swanlab_local
## train
```
HF_HUB_OFFLINE=1 lerobot-train \
  --dataset.repo_id=local/uqi_teleop_data_0514 \
  --dataset.root=datasets/uqi_teleop_data_0514_lerobot \
  --dataset.video_backend=torchcodec \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/diffusion_uqi_teleop_0514_bs8_50k_seed1000_v1 \
  --job_name=diffusion_uqi_teleop_0514_bs8_50k_seed1000_v1 \
  --policy.device=cuda \
  --steps=50000 \
  --batch_size=8 \
  --num_workers=4 \
  --env_eval_freq=0 \
  --save_freq=10000 \
  --wandb.enable=false \
  --swanlab.enable=true \
  --swanlab.mode=local \
  --swanlab.project=uqi_teleop_0514
```

## view loss:

## View train result

## compare all train results

