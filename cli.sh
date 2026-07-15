PYTHONWARNINGS="ignore::DeprecationWarning" \
MUJOCO_GL=egl \
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=my_libero_formal \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/home/ethan/.cache/modelscope/hub/datasets/HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./outputs/libero-formal-bs32 \
  --steps=100000 \
  --batch_size=32 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=5000
  --save_freq=10000