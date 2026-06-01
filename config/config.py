
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/kaggle/input/datasets/ashenafifasilkebede/dataset")
RESULTS_DIR  = Path("/kaggle/working/OralCancerNet/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Magnification experiments ───────────────────────────────────────────────
# Each key is a separate independent experiment.
# MAGNIFICATIONS = ["100x", "400x", "both"] — "both" is for a later ablation
MAGNIFICATIONS = ["100x", "400x"]

# ── Classes ─────────────────────────────────────────────────────────────────
CLASSES       = ["Normal", "OSCC"]
NUM_CLASSES   = 2
CLASS_TO_IDX  = {"Normal": 0, "OSCC": 1}

# ── Image settings ──────────────────────────────────────────────────────────
IMAGE_SIZE    = 224   # all models expect 224×224
CHANNELS      = 3

# ── Training hyperparameters ────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
EARLY_STOPPING_PATIENCE = 10   # stop if val loss doesn't improve for 10 epochs

# ── Model settings ──────────────────────────────────────────────────────────
# Pretrained weights from ImageNet — critical for small datasets like this
PRETRAINED    = True

# VGG19
VGG19_DROPOUT = 0.5

# ViT — using vit_b_16 (base, patch size 16)
VIT_MODEL     = "vit_b_16"

# Swin-T — using swin_t (tiny variant)
SWIN_MODEL    = "swin_t"

# ── Augmentation ─────────────────────────────────────────────────────────────
# Applied only during training, not val/test
# Histopathology-specific: no extreme color jitter (H&E staining is standardized)
AUG_HFLIP_P       = 0.5
AUG_VFLIP_P       = 0.5
AUG_ROTATION_DEG  = 90      # microscopy images have no canonical orientation
AUG_COLOR_JITTER  = 0.2     # slight brightness/contrast variation
AUG_NORMALIZE_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats — fine for H&E
AUG_NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
