# Data

## Shipped with the repository

| Path | Content |
|---|---|
| `annotations/rfw/{internvl,ola,ovis,qwen,sailvl}_labels.csv` | raw predictions of the five VLMs for the 40,607 RFW images |
| `annotations/rfw/labels_RFW.csv` | strict-majority labels (Supp. A.1) after resolving multi-face images: only Supp. Tabs. 2-3 |
| `annotations/rfw/labels_RFW_mode.csv` | plurality-vote labels: used by every experiment (Tab. 1, Figs. 4-5, Tab. 2, Supp. Tabs. 4-5) |
| `annotations/rfw/manual_overrides_{strict,mode}.csv` | manual decisions for images where the VLMs described several faces |
| `annotations/rfw/human_annotations_1k.csv` | human annotations of the 1K validation sample (Supp. A.2), made with `preprocessing/annotation/annotate_rfw_sample.ipynb` |
| `annotations/rfw/human_vs_vlm_confusion.json` | confusion matrices human vs. VLM on the 1K validation sample (Supp. Fig. 1), as stored in the original experiment |
| `rfw_image_order.txt` | the 40,602 RFW images with a detected face, in the order of the embedding files; **all RFW indices in this repository refer to this order** |
| `groups/rfw_groups_ABCD.json` | indices of the groups A-D of Figs. 1-3 and Tab. 2 |
| `celeba_image_order.txt` | the 202,513 CelebA images in the order of the embedding files; all CelebA indices refer to this order |
| `seeds/rfw_kmeanspp_seeds.json` | the 400 k-means++ seeds per backbone used on RFW (k-means++ is not reproducible across scikit-learn versions, so they are shipped) |
| `seeds/celeba_seeds.json` | CelebA seeds of Supp. Figs. 6-12 |
| `celeba_legacy_init/*_100_CelebA_clusters.npy` | initial CelebA clusters used for Supp. Tab. 6 |
| `traversals/classifier_outputs/*.csv.gz` | attribute-classifier probabilities (47 MAAD-Face attributes, `traversals/attribute_classifier`) of every traversed image of Fig. 3 (Arc2Face / InfiniteYou, positive / negative) |
| `traversals/attribute_probabilities.json` | per-strength means plotted in Fig. 3 (`traversals/fig3_aggregate.py`) |

## To download

Set `LATENTALIGN_DATA` (default: this folder) to a directory with:

```
RFW/
  data/{African,Asian,Caucasian,Indian}/<identity>/<image>.jpg    # RFW test images
  embeddings/                                                     # written by preprocessing/embeddings
CelebA/
  img_align_celeba/*.jpg
  identity_CelebA.txt                                             # CelebA identity annotations
  label_cleaned_mso.csv                                           # CelebA attributes (image_id + 40 columns, -1/1)
  embeddings/
```

* RFW: request access at http://www.whdeng.cn/RFW/ .
* CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (aligned images,
  `identity_CelebA.txt`). `label_cleaned_mso.csv`: the cleaned CelebA attributes of Wu et al., *Consistency and accuracy of CelebA attribute values*, CVPRW 2023 ([54] in the paper), https://github.com/HaiyuWu/CelebAMSO .
* Backbones (IResNet-100): ArcFace MS1MV3, CosFace Glint360K and PartialFC
  Glint360K from the [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo),
  ElasticFace from https://github.com/fdbtrs/ElasticFace .
