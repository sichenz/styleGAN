# styleGAN

MUST USE PYTHON VERSION 3.9

### 1. **Using Pre-Trained Models**

#### a. **Generate Images**
Use a pre-trained model to generate images:
```bash
python gen_images.py --outdir=out --trunc=1 --seeds=2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```
- Outputs will be saved in the `out/` directory.

#### b. **Create Interpolation Video**
Render a grid of interpolated images:
```bash
python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

#### c. **Interactive Visualization**
Launch the interactive visualizer to explore a model's properties:
```bash
python visualizer.py
```

---

### 2. **Training New Models**

#### a. **Prepare Dataset**
Use `dataset_tool.py` to prepare datasets in ZIP format:
```bash
python dataset_tool.py --source=/path/to/images --dest=~/datasets/mydataset-1024x1024.zip
```
NOTE: image dimensions MUST BE SQUARE

#### b. **Train a Model**
Start training with your dataset:
```bash
python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/mydataset-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=8.2 --mirror=1
```
- Replace options (`--cfg`, `--data`, `--gpus`, etc.) based on your needs.

#### c. **Fine-Tuning**
Fine-tune a pre-trained model:
```bash
python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/mydataset-1024x1024.zip \
    --gpus=1 --batch=16 --gamma=6.6 --resume=<PRETRAINED_MODEL_URL>
```

---

### 3. **Evaluating Model Performance**

#### a. **Quality Metrics**
Compute metrics for a trained model:
```bash
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip \
    --network=<PRETRAINED_MODEL_URL>
```

#### b. **Spectral Analysis**
Analyze spectral properties of the generator:
```bash
python avg_spectra.py stats --source=~/datasets/ffhq-1024x1024.zip
python avg_spectra.py calc --source=~/datasets/ffhq-1024x1024.zip \
    --dest=tmp/training-data.npz --mean=112.684 --std=69.509
python avg_spectra.py heatmap tmp/training-data.npz
```

---

### 4. **Using Pre-Trained Networks in Python**
Integrate pre-trained networks in your Python code:
```python
import pickle
import torch

with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

z = torch.randn([1, G.z_dim]).cuda()
img = G(z, None)  # Generated image
```

---

### 5. **Using Docker**
Run the generation script in Docker:
```bash
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    stylegan3 \
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

---

### 6. **Info**
- `--outdir`: Output directory for results.
- `--network`: URL or path to pre-trained model.
- `--trunc`: Controls truncation; use lower values for more realistic images.
- `--seeds`: Random seeds for reproducibility.
- `--grid`: Specifies grid size (e.g., `4x2`).

---

