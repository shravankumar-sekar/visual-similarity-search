
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Sampler



MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def stratified_split(labels, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)

        n_train = int(0.8 * len(idx))
        n_val = int(0.1 * len(idx))

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    return tuple(
        np.asarray(idx, dtype=np.int64)
        for idx in (train_idx, val_idx, test_idx)
    )


def get_transforms(training=False):
    ops = []

    if training:
        ops.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return transforms.Compose(ops)


class CIFAR100Subset(Dataset):
    def __init__(self, images, labels, indices, training=False):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.transform = get_transforms(training)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = int(self.indices[idx])
        image = Image.fromarray(self.images[original_idx])
        label = int(self.labels[original_idx])

        return self.transform(image), label

class ClassBalancedBatchSampler(Sampler):
    def __init__(self, labels, indices, classes_per_batch=8,
                 samples_per_class=4, seed=42):
        labels = np.asarray(labels)
        indices = np.asarray(indices, dtype=np.int64)

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0

        if classes_per_batch < 2 or samples_per_class < 2:
            raise ValueError(
                "At least 2 classes and 2 samples per class are required"
            )

        self.class_to_indices = {}

        for cls in np.unique(labels[indices]):
            positions = np.flatnonzero(labels[indices] == cls)

            if len(positions) >= samples_per_class:
                self.class_to_indices[int(cls)] = positions

        self.classes = np.array(list(self.class_to_indices))

        if len(self.classes) < classes_per_batch:
            raise ValueError("Not enough eligible classes")

        self.num_batches = len(indices) // (
            classes_per_batch * samples_per_class
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        for _ in range(self.num_batches):
            selected_classes = rng.choice(
                self.classes,
                size=self.classes_per_batch,
                replace=False,
            )

            batch = []

            for cls in selected_classes:
                positions = rng.choice(
                    self.class_to_indices[int(cls)],
                    size=self.samples_per_class,
                    replace=False,
                )

                batch.extend(positions.tolist())

            rng.shuffle(batch)
            yield batch
