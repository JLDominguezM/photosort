import numpy as np


class FaceEngine:
    def __init__(self, device: str = "cuda"):
        import insightface
        providers = (
            ["CUDAExecutionProvider"] if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_s", providers=providers
        )
        ctx_id = 0 if device == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_faces(self, image_path: str) -> list[dict]:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return []
        faces = self.app.get(img)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            results.append({
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w": int(x2 - x),
                "bbox_h": int(y2 - y),
                "embedding": face.embedding,
            })
        return results

    @staticmethod
    def cluster_faces(embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> list[int]:
        from sklearn.cluster import DBSCAN
        if len(embeddings) < 2:
            return list(range(len(embeddings)))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        distance_matrix = 1 - normalized @ normalized.T
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(distance_matrix)
        return labels.tolist()

    @staticmethod
    def compute_centroid(embeddings: np.ndarray) -> np.ndarray | None:
        """Return the L2-normalized mean of a set of face embeddings, or
        None if the input is empty."""
        if embeddings is None or len(embeddings) == 0:
            return None
        mean = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean.astype(np.float32)

    @staticmethod
    def match_face(embedding: np.ndarray, centroids: dict[int, np.ndarray], threshold: float = 0.6) -> int | None:
        best_id = None
        best_score = -1
        emb_norm = embedding / (np.linalg.norm(embedding) or 1)
        for person_id, centroid in centroids.items():
            c_norm = centroid / (np.linalg.norm(centroid) or 1)
            score = float(emb_norm @ c_norm)
            if score > best_score:
                best_score = score
                best_id = person_id
        if best_score >= threshold:
            return best_id
        return None
