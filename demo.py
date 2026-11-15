
"""Streamlit visual similarity search demo for CIFAR-100."""

import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from data import get_transforms
from model import EmbeddingNet
from train import load_cifar100, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "stage6_random" / "best.pt"
)

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle",
    "mountain", "mouse", "mushroom", "oak_tree",
    "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate",
    "poppy", "porcupine", "possum", "rabbit", "raccoon",
    "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]

assert len(CIFAR100_CLASSES) == 100


def class_name(label):
    """Return the official CIFAR-100 fine-class name."""
    index = int(label)

    if not 0 <= index < len(CIFAR100_CLASSES):
        raise ValueError(f"Invalid CIFAR-100 label: {index}")

    return CIFAR100_CLASSES[index].replace("_", " ")


def search_top_k(query_embedding, gallery_embeddings, k=5):
    """Find the nearest embeddings using exact NumPy L2 search.

    Returns:
        indices: positions of nearest gallery images.
        distances: Euclidean distances in ascending order.

    This function intentionally uses no FAISS or approximate index.
    """
    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    )
    gallery = np.asarray(
        gallery_embeddings,
        dtype=np.float32,
    )

    if query.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional")

    if gallery.ndim != 2:
        raise ValueError("Gallery embeddings must be two-dimensional")

    if gallery.shape[1] != query.shape[0]:
        raise ValueError("Query and gallery dimensions differ")

    if not 1 <= k <= len(gallery):
        raise ValueError("Invalid number of neighbors")

    if (
        not np.isfinite(query).all()
        or not np.isfinite(gallery).all()
    ):
        raise ValueError("Embeddings contain non-finite values")

    difference = gallery - query[None, :]

    squared_distances = np.sum(
        difference * difference,
        axis=1,
    )

    # Stable sorting produces deterministic tie handling.
    indices = np.argsort(
        squared_distances,
        kind="stable",
    )[:k]

    distances = np.sqrt(
        np.maximum(squared_distances[indices], 0.0)
    )

    return indices, distances


def get_normalization_parameters():
    """Find the Normalize operation in evaluation transforms."""
    pipeline = get_transforms(training=False)

    operations = getattr(
        pipeline,
        "transforms",
        [pipeline],
    )

    for operation in operations:
        if isinstance(operation, transforms.Normalize):
            mean = torch.tensor(
                operation.mean,
                dtype=torch.float32,
            ).reshape(3, 1, 1)

            std = torch.tensor(
                operation.std,
                dtype=torch.float32,
            ).reshape(3, 1, 1)

            return mean, std

    return None


def tensor_to_image(tensor):
    """Convert an evaluation tensor back to a displayable image."""
    image = tensor.detach().cpu().float()

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a CHW RGB tensor")

    normalization = get_normalization_parameters()

    if normalization is not None:
        mean, std = normalization
        image = image * std + mean

    image = image.clamp(0.0, 1.0)

    array = (
        image.permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)

    return Image.fromarray(array, mode="RGB")


def load_model(checkpoint_path):
    """Load the trained 30-epoch model without an optimizer."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing trained checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    configuration = checkpoint["config"]

    dimension = int(
        configuration.get("embedding_dim", 128)
    )

    if dimension != 128:
        raise ValueError(
            "This demo expects the trained 128D baseline"
        )

    model = EmbeddingNet(embedding_dim=dimension)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.eval()

    return model, checkpoint


@st.cache_resource(show_spinner="Preparing test gallery...")
def load_search_engine(checkpoint_string):
    """Build gallery embeddings once per Streamlit process."""
    torch.set_num_threads(8)
    set_seed(42)

    checkpoint_path = Path(checkpoint_string)

    model, checkpoint = load_model(checkpoint_path)

    _, _, test_loader = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )

    test_dataset = test_loader.dataset

    if len(test_dataset) != 6000:
        raise RuntimeError(
            f"Expected 6000 test images, got {len(test_dataset)}"
        )

    gallery_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    embeddings_list = []
    labels_list = []

    with torch.inference_mode():
        for images, labels in gallery_loader:
            outputs = model(images)

            if outputs.ndim != 2 or outputs.shape[1] != 128:
                raise RuntimeError(
                    "Model produced unexpected embeddings"
                )

            embeddings_list.append(
                outputs.cpu().numpy().astype(np.float32)
            )
            labels_list.append(
                labels.cpu().numpy()
            )

    gallery_embeddings = np.concatenate(
        embeddings_list,
        axis=0,
    )

    gallery_labels = np.concatenate(
        labels_list,
        axis=0,
    )

    if gallery_embeddings.shape != (6000, 128):
        raise RuntimeError(
            "Unexpected gallery embedding dimensions"
        )

    norms = np.linalg.norm(
        gallery_embeddings,
        axis=1,
    )

    if not np.allclose(norms, 1.0, atol=0.001):
        raise RuntimeError(
            "Gallery embeddings are not L2-normalized"
        )

    return (
        model,
        checkpoint,
        test_dataset,
        gallery_embeddings,
        gallery_labels,
    )


def embed_uploaded_image(model, pil_image):
    """Embed an uploaded RGB image using evaluation transforms."""
    image = pil_image.convert("RGB")

    # The network was trained on 32x32 CIFAR-100 images.
    image = image.resize(
        (32, 32),
        Image.Resampling.BILINEAR,
    )

    preprocessing = get_transforms(training=False)

    tensor = preprocessing(image).unsqueeze(0)

    with torch.inference_mode():
        embedding = model(tensor)[0]

    result = embedding.cpu().numpy().astype(np.float32)

    if result.shape != (128,):
        raise RuntimeError(
            "Expected a 128-dimensional query embedding"
        )

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Query embedding contains non-finite values"
        )

    return result


def main():
    st.set_page_config(
        page_title="Visual Similarity Search",
        page_icon="🔎",
        layout="wide",
    )

    st.title("Visual Similarity Search")
    st.write(
        "Find visually similar images using a scratch-trained "
        "ResNet-18 embedding model and exact NumPy L2 retrieval."
    )

    try:
        (
            model,
            checkpoint,
            test_dataset,
            gallery_embeddings,
            gallery_labels,
        ) = load_search_engine(str(CHECKPOINT_PATH))

    except Exception as error:
        st.error(
            f"Could not initialize the search engine: {error}"
        )
        st.stop()

    configuration = checkpoint["config"]

    with st.sidebar:
        st.header("Model information")
        st.write("**Dataset:** CIFAR-100")
        st.write("**Backbone:** ResNet-18")
        st.write("**Pretrained weights:** None")
        st.write("**Embedding dimension:** 128")
        st.write("**Gallery size:** 6,000 images")
        st.write(
            f"**Checkpoint epoch:** {checkpoint['epoch']}"
        )
        st.write(
            f"**Triplet mining:** "
            f"{configuration.get('mining', 'unknown')}"
        )
        st.write(
            f"**Margin:** "
            f"{configuration.get('margin', 'unknown')}"
        )
        st.write(
            f"**Optimizer:** "
            f"{configuration.get('optimizer', 'unknown')}"
        )
        st.caption(
            "Inference uses CPU. No pretrained model or "
            "external vector-search service is used."
        )

    st.subheader("Upload a query image")

    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, JPEG or WEBP image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to see its five nearest "
            "CIFAR-100 test-gallery matches."
        )
        return

    try:
        query_image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")
    except Exception as error:
        st.error(f"Cannot read image: {error}")
        return

    left, right = st.columns([1, 3])

    with left:
        st.image(
            query_image,
            caption="Uploaded query image",
            use_container_width=True,
        )

    try:
        query_embedding = embed_uploaded_image(
            model,
            query_image,
        )

        indices, distances = search_top_k(
            query_embedding,
            gallery_embeddings,
            k=5,
        )

    except Exception as error:
        st.error(f"Search failed: {error}")
        return

    with right:
        st.subheader("Top-5 nearest matches")
        st.write(
            "Results are ranked by ascending Euclidean "
            "distance in the learned 128D embedding space."
        )

        columns = st.columns(5)

        for rank, (index, distance) in enumerate(
            zip(indices, distances),
            start=1,
        ):
            gallery_tensor, dataset_label = test_dataset[
                int(index)
            ]

            expected_label = int(
                gallery_labels[int(index)]
            )

            if int(dataset_label) != expected_label:
                st.error(
                    "Gallery label mismatch; results cannot "
                    "be displayed reliably."
                )
                st.stop()

            gallery_image = tensor_to_image(
                gallery_tensor
            )

            # This is a display-only monotonic score,
            # not a calibrated probability.
            similarity_score = 1.0 / (
                1.0 + float(distance)
            )

            with columns[rank - 1]:
                st.image(
                    gallery_image,
                    use_container_width=True,
                )

                st.markdown(
                    f"**#{rank} — "
                    f"{class_name(expected_label)}**"
                )

                st.caption(
                    f"L2 distance: {distance:.4f}"
                )

                st.caption(
                    f"Similarity: {similarity_score:.4f}"
                )

        st.caption(
            "Similarity = 1 / (1 + Euclidean distance). "
            "Higher is better. It is not a probability."
        )

    st.divider()

    st.info(
        "The model was trained on CIFAR-100 images at "
        "32 × 32 resolution. Uploading unrelated photographs "
        "may produce weak or misleading matches. "
        "The search is image similarity, not general-purpose "
        "object recognition."
    )


if __name__ == "__main__":
    main()
