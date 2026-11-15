# Visual Similarity Search using Deep Metric Learning

A visual similarity search project built with PyTorch, NumPy and Streamlit using the CIFAR-100 dataset.

The goal was to train an image embedding model from scratch and use those embeddings to retrieve visually similar images from a gallery. I also implemented the triplet loss, negative mining methods, optimizers and nearest-neighbor search rather than relying on metric-learning or vector-search libraries.

The project was developed and tested in Ubuntu WSL on a CPU-only machine.

## 1. Approach

I used a ResNet-18 backbone without pretrained weights. The final classification layer was replaced with a linear projection to generate 128-dimensional embeddings, followed by L2 normalization.

For training, I used triplet loss to encourage images from the same class to have smaller embedding distances than images from different classes.

The retrieval process is straightforward:

1. Pass an image through the trained model.
2. Extract its 128-dimensional embedding.
3. Calculate its Euclidean distance to the gallery embeddings.
4. Sort the distances.
5. Return the five nearest images with their class labels and similarity scores.

The main components are implemented in separate Python files so that training, evaluation and retrieval can be tested independently.

## 2. Dataset

I used CIFAR-100, which contains 60,000 images across 100 classes.

The dataset was split using a fixed random seed of 42.

| Split | Images |
|---|---:|
| Training | 48,000 |
| Validation | 6,000 |
| Test | 6,000 |

The split was stratified so that each class contributed the same proportion of images.

Training batches contained 16 classes with four images per class, giving a batch size of 64.

The validation split was used for model selection and ablation experiments. The test split was kept separate for the final evaluation.

## 3. Implementation

### Model

The embedding model is defined in `model.py`.

I used ResNet-18 with random initialization and modified the input stem for CIFAR-100's 32×32 images. The final layer produces L2-normalized embeddings.

### Triplet loss and mining

I implemented triplet loss in `loss.py` and experimented with three mining strategies:

- Random triplet mining
- Batch-hard mining
- Semi-hard mining

One of the main problems I encountered was embedding collapse when training with batch-hard mining from scratch.

The model started producing embeddings with very little variation, which made image retrieval ineffective.

I investigated this by measuring embedding spread and comparing the behavior of different mining strategies. Random mining gave me stable training, so I used it for the main 30-epoch experiment.

Semi-hard mining also performed well in shorter experiments.

### Optimizers

I implemented Momentum SGD, Adam and NAdam using PyTorch tensor operations without using `torch.optim` for the optimizer updates.

The implementations are in `optimizers.py`.

I tested all three under the same initial experimental settings.

### Retrieval

The retrieval implementation is in `retrieval.py`.

It uses NumPy to calculate Euclidean distances and retrieve the nearest images from the gallery.

I used exact brute-force search rather than FAISS or another vector database.

## 4. Training

The main model was trained for 30 complete epochs.

| Parameter | Value |
|---|---|
| Backbone | ResNet-18 |
| Embedding dimension | 128 |
| Optimizer | Custom Adam |
| Learning rate | 0.0001 |
| Triplet margin | 0.3 |
| Mining strategy | Random |
| Training batch size | 64 |
| Epochs | 30 |
| Random seed | 42 |

Training was performed on CPU with eight PyTorch threads.

I saved checkpoints during training and selected the best model using validation mAP.

To reproduce the main experiment:

```bash
python -u scripts/train_full.py \
  --epochs 30 \
  --threads 8 \
  --lr 0.0001 \
  --margin 0.3 \
  --seed 42 \
  --run-dir checkpoints/stage6_random
```

To resume an interrupted run, use the same command with `--resume`.

## 5. Evaluation Results

After training, I evaluated the model on the held-out CIFAR-100 test split.

The retrieval evaluation used 500 queries selected across the 100 classes and searched against the full test gallery.

| Metric | Result |
|---|---:|
| Recall@1 | 0.481% |
| Recall@5 | 2.193% |
| Recall@10 | 3.997% |
| Hit@1 | 28.4% |
| Hit@5 | 57.0% |
| Hit@10 | 67.4% |
| mAP | 15.525% |

I also evaluated triplet ordering using 5,000 randomly sampled triplets.

**Random-triplet ordering accuracy: 86.98%.**

This exceeded the assignment's 75% target under the random-triplet evaluation protocol. However, the model did not meet that threshold when using the hardest negative from the entire gallery.

The results show that the model learned some useful image similarities, but retrieval performance still has considerable room for improvement.

Run the test evaluation with:

```bash
python -u scripts/evaluate_test.py \
  --checkpoint checkpoints/stage6_random/best.pt \
  --threads 8 \
  --queries-per-class 5 \
  --triplets-per-query 10 \
  --margin 0.3 \
  --output results/stage6_test.json \
  --csv results/stage6_test.csv
```

## 6. Ablation Experiments

I ran ablation experiments to understand how embedding dimension, triplet margin, optimizer and mining strategy affected retrieval performance.

For the controlled comparisons, I changed one parameter at a time and trained each configuration for three epochs using the same data split and random seed.

### Embedding dimension

| Dimension | Validation mAP |
|---|---:|
| 64 | 4.710% |
| 128 | 4.558% |
| 256 | 4.837% |

256 dimensions gave the highest mAP after three epochs, although the difference was small.

### Triplet margin

| Margin | Validation mAP |
|---|---:|
| 0.1 | 4.884% |
| 0.3 | 4.558% |
| 0.5 | 4.641% |

Margin 0.1 performed best in this comparison.

### Optimizers

| Optimizer | Validation mAP |
|---|---:|
| Momentum SGD | 2.776% |
| Adam | 4.558% |
| NAdam | 4.612% |

Adam and NAdam gave similar results. Momentum SGD performed worse at the common learning rate of 0.0001.

This comparison does not establish which optimizer would perform best after separate learning-rate tuning.

### Mining strategies

| Strategy | Epochs completed | Validation mAP |
|---|---:|---:|
| Random | 3 | 4.558% |
| Batch-hard | 1 | 1.903% |
| Semi-hard | 3 | 5.015% |

Batch-hard mining caused embedding collapse and training was stopped after the first epoch.

Semi-hard mining achieved the best three-epoch mAP, while random mining was the strategy used for the successful 30-epoch baseline.

I did not combine the individual ablation winners into a new long training run, so their longer-term performance remains unknown.

The complete results, including Recall@K, are available in [the ablation report](docs/ablation_report.md).

## 7. Streamlit Demo

I built a small Streamlit application to test the trained model interactively.

The user can upload an image, and the application returns the five nearest images from the CIFAR-100 test gallery.

Each retrieved image is displayed with its class label, Euclidean distance and a similarity score derived from the distance.

To launch the application:

```bash
python -m streamlit run demo.py
```

Open `http://localhost:8501` in a browser.

The application uses the checkpoint at:

```text
checkpoints/stage6_random/best.pt
```

This checkpoint is generated by training and is not included in the Git repository.

The similarity score shown in the application is calculated as `1 / (1 + distance)`. It is only a convenient display score, not a probability.

### Demo recording

I recorded five queries using the Streamlit interface to demonstrate the retrieval process.

**Video:** [Watch the Streamlit demo](https://drive.google.com/file/d/1GQ6iPCIp5cjUK7GOo1_A1EKUppnSZ_TF/view?usp=sharing)

## 8. Project Structure

```text
visual-similarity-search/
├── data.py
├── epoch_sampler.py
├── model.py
├── loss.py
├── optimizers.py
├── random_mining.py
├── semi_hard_mining.py
├── retrieval.py
├── train.py
├── demo.py
├── scripts/
│   ├── train_full.py
│   ├── evaluate_test.py
│   ├── run_ablation.py
│   └── generate_ablation_report.py
├── tests/
├── docs/
│   └── ablation_report.md
├── requirements.txt
└── README.md
```

Additional diagnostic and evaluation scripts are also available in the `scripts/` directory.

## 9. Setup

The project was developed using Python 3.14.4 on Ubuntu WSL with CPU-only PyTorch.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/shravankumar-sekar/visual-similarity-search.git
cd visual-similarity-search

python3 -m venv .venv
source .venv/bin/activate
```

The development environment used PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu. Install compatible CPU builds for your Python version using the official PyTorch instructions.

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The current dependency file reflects the development environment. Installation on a fresh machine should be verified, particularly for the CPU-specific PyTorch builds.

The CIFAR-100 dataset is downloaded using torchvision when required.

To run the demo without an existing checkpoint, first complete model training using the command in Section 4.

## 10. Testing

I wrote automated tests for the dataset pipeline, sampling, model, loss functions, optimizers, checkpoint handling, retrieval evaluation, ablation experiments and Streamlit search logic.

Run:

```bash
python -m pytest tests/ -q
```

Latest local result:

```text
161 passed in 10.25s
```

The test result was obtained in the existing WSL development environment.

## 11. Challenges and Limitations

The most difficult issue was getting triplet-based training to remain stable.

My initial batch-hard mining experiments resulted in embedding collapse. I tested alternative mining methods and monitored embedding spread to understand what was happening.

Random mining worked reliably for the full training run, but using it means the final model does not fully satisfy the assignment's specified batch-hard training configuration.

Other limitations include:

- The short ablation experiments used only one random seed and three training epochs.
- The model was trained on small CIFAR-100 images and does not generalize reliably to arbitrary photographs.
- The nearest retrieved image is not always from the correct class.
- The model achieved the random-triplet ordering target but not the strict hardest-negative ordering criterion.
- Training was CPU-only, which restricted how many longer experiments I could run.

These are areas I would investigate further rather than treating the current model as production-ready.

## 12. Conclusion

I built and evaluated a visual similarity search pipeline using a ResNet-18 model trained from scratch.

The project covers dataset preparation, deep metric learning, custom optimizers, negative mining, checkpointing, retrieval evaluation, ablation experiments and an interactive application.

The main 30-epoch model achieved **15.525% test mAP** and **86.98% random-triplet ordering accuracy**.

The biggest technical takeaway was how strongly negative mining affected training stability. Batch-hard mining caused collapse under the tested settings, while random and semi-hard mining allowed the model to learn more useful embeddings.

The project helped me understand not only how to implement an image retrieval system, but also how to investigate training failures and evaluate whether the learned embeddings are useful.