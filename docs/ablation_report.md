# Visual Similarity Search - Ablation Report

## Evaluation protocol

- Dataset: CIFAR-100, stratified seed-42 split.
- Training: 48,000 images per complete epoch.
- Validation: 6,000 gallery images, 500 queries.
- Model: ResNet-18 trained from scratch.
- Baseline: 128D, margin 0.3, custom Adam, random mining.
- Batch size: 64 (16 classes x 4 examples).
- Learning rate: 0.0001.
- Standard budget: three training epochs.
- Checkpoints selected by validation mAP.
- Retrieval uses brute-force Euclidean distance.
- Query self-matches are excluded.

Recall@K is the fraction of relevant gallery images retrieved within the first K results. Hit@K is the fraction of queries with at least one relevant image in the top K. mAP uses the complete gallery ranking.

## Embedding dimension

| Configuration | Epochs | Recall@1 | Recall@5 | Recall@10 | Hit@1 | Hit@5 | Hit@10 | mAP | Collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64D | 3 | 0.180% | 0.739% | 1.298% | 10.600% | 29.200% | 40.400% | 4.710% | No |
| 128D | 3 | 0.173% | 0.729% | 1.210% | 10.200% | 28.600% | 39.200% | 4.558% | No |
| 256D | 3 | 0.197% | 0.807% | 1.380% | 11.600% | 30.600% | 42.000% | 4.837% | No |

**Findings:**

- 64D: 4.710% mAP, +0.152 percentage points relative to the 128D control.
- 256D: 4.837% mAP, +0.279 percentage points relative to the 128D control.

## Triplet margin

| Configuration | Epochs | Recall@1 | Recall@5 | Recall@10 | Hit@1 | Hit@5 | Hit@10 | mAP | Collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.1 | 3 | 0.173% | 0.725% | 1.366% | 10.200% | 30.400% | 40.800% | 4.884% | No |
| 128D | 3 | 0.173% | 0.729% | 1.210% | 10.200% | 28.600% | 39.200% | 4.558% | No |
| 0.5 | 3 | 0.183% | 0.614% | 1.217% | 10.800% | 27.200% | 41.000% | 4.641% | No |

**Findings:**

- 0.1: 4.884% mAP, +0.326 percentage points relative to the 128D control.
- 0.5: 4.641% mAP, +0.083 percentage points relative to the 128D control.

## Optimizer

| Configuration | Epochs | Recall@1 | Recall@5 | Recall@10 | Hit@1 | Hit@5 | Hit@10 | mAP | Collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Momentum SGD | 3 | 0.112% | 0.451% | 0.756% | 6.600% | 19.800% | 27.800% | 2.776% | No |
| 128D | 3 | 0.173% | 0.729% | 1.210% | 10.200% | 28.600% | 39.200% | 4.558% | No |
| NAdam | 3 | 0.173% | 0.658% | 1.173% | 10.200% | 27.000% | 37.000% | 4.612% | No |

**Findings:**

- Momentum SGD: 2.776% mAP, -1.782 percentage points relative to the 128D control.
- NAdam: 4.612% mAP, +0.055 percentage points relative to the 128D control.

## Mining strategy

| Configuration | Epochs | Recall@1 | Recall@5 | Recall@10 | Hit@1 | Hit@5 | Hit@10 | mAP | Collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128D | 3 | 0.173% | 0.729% | 1.210% | 10.200% | 28.600% | 39.200% | 4.558% | No |
| Batch-hard | 1 | 0.088% | 0.356% | 0.641% | 5.200% | 17.200% | 27.000% | 1.903% | Yes |
| Semi-hard | 3 | 0.166% | 0.766% | 1.366% | 9.800% | 30.600% | 42.000% | 5.015% | No |

**Findings:**

- Batch-hard collapsed after 1 epoch(s); this is not a matched three-epoch result.
- Semi-hard: 5.015% mAP, +0.457 percentage points relative to the 128D control.

## Collapse diagnostic

| Configuration | Spread | Spread ratio | Collapsed |
|---|---:|---:|---|
| 64D | 0.569793 | 4.854 | No |
| 128D | 0.558079 | 4.217 | No |
| 256D | 0.575661 | 4.423 | No |
| 0.1 | 0.199255 | 1.506 | No |
| 0.5 | 0.861415 | 6.509 | No |
| Momentum SGD | 0.566925 | 4.284 | No |
| NAdam | 0.557859 | 4.216 | No |
| Batch-hard | 0.013748 | 0.104 | Yes |
| Semi-hard | 0.979877 | 7.405 | No |

## Independent 30-epoch baseline test results

The main model used 128D embeddings, margin 0.3, custom Adam and random mining.

| Test metric | Result |
|---|---:|
| Recall@1 | 0.481% |
| Recall@5 | 2.193% |
| Recall@10 | 3.997% |
| Hit@1 | 28.400% |
| Hit@5 | 57.000% |
| Hit@10 | 67.400% |
| mAP | 15.525% |
| Random-triplet ordering accuracy | 86.980% |

The held-out test numbers were measured separately and are not recalculated by this report generator.

## Limitations

1. Ablations used one seed and only three epochs.
2. Optimizer learning rates were not individually tuned.
3. Short-run winners were not tested together.
4. Batch-hard mining collapsed after its first epoch.
5. The final 30-epoch model used random mining, not batch-hard mining.
6. Random-triplet ordering accuracy is not the same as hard-negative ordering accuracy.

## Conclusion

Semi-hard mining had the strongest short-run mining mAP; batch-hard mining collapsed. The stable 30-epoch random-mining baseline achieved 15.525% held-out test mAP.
