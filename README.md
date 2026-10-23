# PS-18: Breast Cancer Histopathology Classification (BreakHis)

**Name:** Barathraj R | **Register No.:** RA2311026050069 | **Section:** AIML B

Binary classification (benign vs malignant) of breast tumour histopathology images from BreakHis at four magnifications (40x, 100x, 200x, 400x). The project studies two questions:

1. Does accuracy change with magnification?
2. What is the effect of the same patient appearing in both training and test sets?

## Dataset

- BreakHis: 7,909 images (2,480 benign, 5,429 malignant), 4 magnifications, 81 patient IDs in the copy used.
- Source: Hugging Face mirror `hirundo-io/BreaKHis-original-9` (`breakhis.zip`).
- Patient ID, class and magnification are parsed from file names (e.g. `SOB_B_F-14-23060AB-400-002.jpg`).

## Method

- ImageNet-pretrained **ResNet18**, fine-tuned end to end with a 2-class output layer (224 x 224 input, mixed precision).
- Two 5-fold cross-validation schemes with identical model and settings:
  - **Patient-wise:** `StratifiedGroupKFold` grouped by patient (zero patient overlap between train and test).
  - **Image-level (leaky):** `StratifiedKFold` over images (all 81 patients appear in both train and test).
- Metrics: accuracy, balanced accuracy, F1, AUC, sensitivity, specificity, with patient-level bootstrap 95% CIs (1,000 resamples).

## Results (pooled out-of-fold predictions)

| Split | Mag. | Accuracy | 95% CI | Bal. acc. | F1 | AUC | Sens. | Spec. |
|---|---|---|---|---|---|---|---|---|
| Patient-wise | All | 0.864 | - | 0.849 | 0.900 | 0.925 | 0.890 | 0.807 |
| Patient-wise | 40x | 0.859 | 0.801-0.912 | 0.833 | 0.898 | 0.918 | 0.902 | 0.763 |
| Patient-wise | 100x | 0.877 | 0.830-0.920 | 0.864 | 0.909 | 0.933 | 0.897 | 0.831 |
| Patient-wise | 200x | 0.873 | 0.817-0.922 | 0.863 | 0.906 | 0.929 | 0.888 | 0.838 |
| Patient-wise | 400x | 0.847 | 0.789-0.898 | 0.834 | 0.885 | 0.914 | 0.872 | 0.796 |
| Image-level (leaky) | All | 0.982 | - | 0.983 | 0.987 | 0.998 | 0.980 | 0.986 |
| Image-level (leaky) | 40x | 0.986 | 0.981-0.992 | 0.986 | 0.990 | 0.998 | 0.988 | 0.984 |
| Image-level (leaky) | 100x | 0.984 | 0.975-0.991 | 0.986 | 0.988 | 0.998 | 0.982 | 0.989 |
| Image-level (leaky) | 200x | 0.982 | 0.971-0.991 | 0.984 | 0.987 | 0.997 | 0.980 | 0.987 |
| Image-level (leaky) | 400x | 0.974 | 0.957-0.987 | 0.977 | 0.980 | 0.996 | 0.968 | 0.985 |

Patient-level accuracy (average probability per patient): **0.901** patient-wise, 1.000 image-level.

## Key findings

- **Magnification:** patient-wise accuracy ranges from 0.847 (400x) to 0.877 (100x). The gap is smaller than the fold standard deviation and the 95% CIs overlap, so no reliable magnification effect can be claimed with 81 patients.
- **Patient leakage:** putting the same patients in train and test inflates accuracy from 0.864 to 0.982 (+11.7 points) and makes the estimate look falsely precise (fold SD 0.035 -> 0.005). The patient-wise number is the realistic one.

## Files

| File | Description |
|---|---|
| `PS18_BreakHis.ipynb` | Full notebook (data loading, training, evaluation, plots) |
| `resnet18_breakhis_patientwise.pth` | ResNet18 weights (fold 1, patient-wise split) |
| `results.csv` | Per-magnification metrics for both splits |
| `PS18_Report.pdf` | Project report |

## How to run

1. Open the notebook in Google Colab and set the runtime to a GPU (Runtime -> Change runtime type -> T4 GPU).
2. Run all cells. The dataset downloads automatically from Hugging Face:

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("hirundo-io/BreaKHis-original-9", "breakhis.zip", repo_type="dataset")
```

3. To load the saved model:

```python
import torch, torchvision
m = torchvision.models.resnet18()
m.fc = torch.nn.Linear(m.fc.in_features, 2)
m.load_state_dict(torch.load("resnet18_breakhis_patientwise.pth", map_location="cpu"))
m.eval()
```

## Limitations

Only 81 patients from a single laboratory, no external test set, one backbone, and an untuned 0.5 threshold. This is a student project and is not for clinical use.

## Reference

Spanhol F.A., Oliveira L.S., Petitjean C., Heutte L., "A Dataset for Breast Cancer Histopathological Image Classification", IEEE Trans. Biomed. Eng. 63(7):1455-1462 (2016).
