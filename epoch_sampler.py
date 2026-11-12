
import random

from torch.utils.data import Sampler


class EpochBalancedBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        indices,
        classes_per_batch=8,
        samples_per_class=4,
        seed=42,
    ):
        if classes_per_batch < 2 or samples_per_class < 2:
            raise ValueError("Batch configuration must be at least 2x2")

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0

        self.class_positions = {}

        for position, original_index in enumerate(indices):
            label = int(labels[original_index])
            self.class_positions.setdefault(label, []).append(position)

        self.classes = sorted(self.class_positions)

        if classes_per_batch > len(self.classes):
            raise ValueError("Not enough classes for a batch")

        counts = [
            len(positions)
            for positions in self.class_positions.values()
        ]

        if len(set(counts)) != 1:
            raise ValueError("Classes must have equal sample counts")

        if counts[0] % samples_per_class != 0:
            raise ValueError(
                "Class size must divide evenly by samples_per_class"
            )

        self.rounds = counts[0] // samples_per_class
        self.num_slots = len(self.classes) * self.rounds

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")

        self.epoch = epoch

    def __len__(self):
        return (
            self.num_slots + self.classes_per_batch - 1
        ) // self.classes_per_batch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        positions = {}

        for label, values in self.class_positions.items():
            positions[label] = values.copy()
            rng.shuffle(positions[label])

        class_order = self.classes.copy()
        rng.shuffle(class_order)

        offsets = {label: 0 for label in self.classes}
        slots = class_order * self.rounds

        for start in range(0, len(slots), self.classes_per_batch):
            selected_classes = slots[
                start:start + self.classes_per_batch
            ]

            batch = []

            for label in selected_classes:
                offset = offsets[label]

                batch.extend(
                    positions[label][
                        offset:offset + self.samples_per_class
                    ]
                )

                offsets[label] += self.samples_per_class

            rng.shuffle(batch)
            yield batch
