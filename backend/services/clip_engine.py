import numpy as np
import torch
import open_clip
import yaml
from PIL import Image
import pillow_heif

from services.logging_config import get_logger

pillow_heif.register_heif_opener()

log = get_logger(__name__)


class CLIPEngine:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(device).eval()
        self.categories: list[dict] = []
        self.category_embeddings: np.ndarray | None = None
        self.threshold: float = 0.22
        self.batch_size: int = 32

    def load_categories(self, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.categories = cfg.get("categories", [])
        self.threshold = cfg.get("threshold", 0.22)
        self.batch_size = cfg.get("batch_size", 32)

        if not self.categories:
            return

        all_embeddings = []
        for cat in self.categories:
            prompts = cat["prompts"]
            tokens = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                text_emb = self.model.encode_text(tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            avg_emb = text_emb.mean(dim=0)
            avg_emb = avg_emb / avg_emb.norm()
            all_embeddings.append(avg_emb.cpu().numpy())

        self.category_embeddings = np.stack(all_embeddings)

    def encode_image(self, image_path: str) -> np.ndarray | None:
        try:
            img = Image.open(image_path).convert("RGB")
            return self.encode_pil(img)
        except Exception as e:
            log.warning("CLIP encode_image failed for %s: %s", image_path, e)
            return None

    def encode_pil(self, img: Image.Image) -> np.ndarray | None:
        try:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten()
        except Exception as e:
            log.warning("CLIP encode_pil failed: %s", e)
            return None

    def encode_video(self, video_path: str, num_frames: int = 3) -> np.ndarray | None:
        """Sample `num_frames` keyframes and return the L2-normalized mean
        embedding as a single vector."""
        from services.video_engine import extract_keyframes

        frames = extract_keyframes(video_path, num_frames=num_frames)
        if not frames:
            return None
        embeddings = [e for e in (self.encode_pil(f) for f in frames) if e is not None]
        if not embeddings:
            return None
        mean = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean.astype(np.float32)

    def encode_images_batch(self, image_paths: list[str]) -> list[np.ndarray | None]:
        results = []
        images = []
        indices = []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.preprocess(img))
                indices.append(i)
            except Exception as e:
                log.warning("CLIP batch skip %s: %s", path, e)
        results = [None] * len(image_paths)
        if not images:
            return results
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            embs = self.model.encode_image(batch)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        embs_np = embs.cpu().numpy()
        for j, idx in enumerate(indices):
            results[idx] = embs_np[j]
        return results

    def classify(self, embedding: np.ndarray) -> tuple[str, float]:
        if self.category_embeddings is None:
            return "Uncategorized", 0.0
        scores = embedding @ self.category_embeddings.T
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < self.threshold:
            return "Uncategorized", best_score
        return self.categories[best_idx]["name"], best_score

    def search(self, query: str, embeddings: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_np = text_emb.cpu().numpy().flatten()
        scores = embeddings @ text_np
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def find_similar(self, target_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        scores = embeddings @ target_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
