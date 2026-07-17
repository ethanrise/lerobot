
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