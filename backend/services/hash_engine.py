from PIL import Image
import imagehash
import pillow_heif

pillow_heif.register_heif_opener()


class HashEngine:
    @staticmethod
    def compute_phash(image_path: str) -> str | None:
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except Exception:
            return None

    @staticmethod
    def find_duplicates(hashes: list[tuple[int, str]], threshold: int = 8) -> list[list[int]]:
        groups = []
        used = set()
        parsed = [(pid, imagehash.hex_to_hash(h)) for pid, h in hashes if h]

        for i, (id_a, hash_a) in enumerate(parsed):
            if id_a in used:
                continue
            group = [id_a]
            for j in range(i + 1, len(parsed)):
                id_b, hash_b = parsed[j]
                if id_b in used:
                    continue
                if hash_a - hash_b <= threshold:
                    group.append(id_b)
                    used.add(id_b)
            if len(group) > 1:
                groups.append(group)
                used.add(id_a)
        return groups
