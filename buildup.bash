conda create -n MaskDiT python==3.10
conda activate MaskDiT

pip install -r requirements.txt

wget https://hzpublic-hub.s3.us-west-2.amazonaws.com/maskdit/checkpoints/imagenet256-ckpt-best_with_guidance.pt
python3 download_assets.py --name vae --dest assets/stable_diffusion

bash scripts/download_assets.sh