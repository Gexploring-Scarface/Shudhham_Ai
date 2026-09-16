import json
import re
from pathlib import Path
from rapidfuzz.fuzz import WRatio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).parent / "knowledge_base.json"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# Broad, explainable ontology. These labels let the system generalize to new phrasings
# without pretending that the knowledge base contains every possible object.
ONTOLOGY = {
    "e_waste": {
        "label": "E-waste",
        "aliases": [
            "laptop", "notebook computer", "computer", "pc", "desktop", "monitor", "screen",
            "keyboard", "mouse", "printer", "scanner", "router", "modem", "mobile phone",
            "smartphone", "phone", "tablet", "charger", "adapter", "earphones", "headphones",
            "earbuds", "power bank", "hard drive", "ssd", "usb drive", "pendrive", "remote",
            "television", "tv", "camera", "game console", "speaker", "electronic device",
            "electronics", "cable", "wire", "iron", "microwave", "toaster", "blender", "mixer",
            "washing machine", "refrigerator", "ac", "air conditioner", "fan", "battery"
        ],
        "keywords": ["electronic", "electrical", "device", "digital", "computer", "phone", "battery", "circuit"]
    },
    "wet": {
        "label": "Wet / Organic",
        "aliases": [
            "banana peel", "fruit peel", "vegetable peel", "food waste", "food scraps", "leftover food",
            "rice", "bread", "vegetable", "fruit", "tea leaves", "tea bag", "coffee grounds", "eggshell",
            "garden waste", "leaves", "grass", "flowers", "plant waste", "organic waste", "compostable waste",
            "spoiled food", "cooked food", "kitchen waste"
        ],
        "keywords": ["organic", "food", "kitchen", "fruit", "vegetable", "peel", "garden", "leaf", "compost"]
    },
    "dry_recyclable": {
        "label": "Dry / Recyclable",
        "aliases": [
            "plastic bag", "polythene bag", "plastic bottle", "pet bottle", "water bottle", "plastic container",
            "plastic packaging", "newspaper", "paper", "cardboard", "carton", "paper box", "magazine",
            "glass bottle", "glass jar", "metal can", "aluminium can", "tin can", "steel can", "plastic wrapper",
            "plastic packet", "wrapping", "packaging", "milk carton", "tissue box", "paper bag", "clean plastic"
        ],
        "keywords": ["plastic", "paper", "cardboard", "glass", "metal", "can", "packaging", "bottle", "carton", "recyclable", "dry"]
    },
    "sanitary": {
        "label": "Sanitary Waste",
        "aliases": ["used sanitary pad", "sanitary pad", "diaper", "used diaper", "adult diaper", "incontinence pad", "used tissue", "bandage", "cotton swab"],
        "keywords": ["sanitary", "pad", "diaper", "hygiene", "medical dressing"]
    },
    "hazardous": {
        "label": "Hazardous Waste",
        "aliases": ["paint", "paint can", "pesticide", "pesticide bottle", "chemical", "solvent", "bleach", "disinfectant", "aerosol can", "insecticide"],
        "keywords": ["hazardous", "chemical", "paint", "pesticide", "solvent", "toxic"]
    },
    "special": {
        "label": "Special Collection",
        "aliases": ["used cooking oil", "cooking oil", "cooking grease", "medicine", "medicines", "tablet", "expired medicine", "pharmaceutical waste", "thermometer", "fluorescent bulb", "led bulb", "light bulb"],
        "keywords": ["oil", "medicine", "pharmaceutical", "bulb", "fluorescent", "special collection"]
    },
    "textile": {
        "label": "Textile / Reuse",
        "aliases": ["old clothes", "clothing", "shirt", "pants", "jeans", "jacket", "towel", "bedsheet", "blanket", "shoes", "footwear", "textile"],
        "keywords": ["cloth", "clothes", "clothing", "textile", "fabric", "shoe", "footwear"]
    },
    "bulky": {
        "label": "Bulky Waste",
        "aliases": ["wooden chair", "chair", "table", "furniture", "mattress", "sofa", "wood", "wooden board", "large plastic furniture"],
        "keywords": ["bulky", "furniture", "wood", "mattress", "sofa", "chair", "table"]
    },
}


class WasteRAG:
    def __init__(self, path=KB_PATH):
        self.docs = json.loads(Path(path).read_text(encoding="utf-8"))
        self.texts = [
            f"{d['item']} {d['keywords']} {d['category']} {d['action']} {d['bin']}"
            for d in self.docs
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.alias_index = self._build_alias_index()

    def _build_alias_index(self):
        pairs = []
        for key, meta in ONTOLOGY.items():
            for alias in meta["aliases"]:
                pairs.append((normalize(alias), key))
        return pairs

    @staticmethod
    def _tokens(text):
        return set(re.findall(r"[a-z0-9]+", normalize(text)))

    def classify_generic(self, query: str):
        """Explainable generic classifier used before retrieval.

        It combines exact alias, token keyword evidence, and fuzzy phrase matching.
        It intentionally returns None when evidence is weak rather than fabricating a category.
        """
        q = normalize(query)
        if not q:
            return None

        # Exact/contains alias gets the strongest signal.
        best = None
        for alias, key in self.alias_index:
            if alias == q or alias in q or q in alias:
                score = 1.0 if alias == q else 0.93
                if best is None or score > best["score"]:
                    best = {"key": key, "matched_alias": alias, "score": score, "method": "semantic alias match"}
        if best:
            return best

        q_tokens = self._tokens(q)
        candidates = []
        for key, meta in ONTOLOGY.items():
            kw = set(meta["keywords"])
            overlap = len(q_tokens & kw)
            keyword_score = overlap / max(1, min(3, len(q_tokens)))
            for alias in meta["aliases"]:
                fuzzy = WRatio(q, normalize(alias)) / 100.0
                combined = 0.58 * fuzzy + 0.42 * keyword_score
                candidates.append((combined, fuzzy, overlap, key, alias))
        candidates.sort(reverse=True)
        score, fuzzy, overlap, key, alias = candidates[0]
        if fuzzy >= 0.86 or (fuzzy >= 0.72 and overlap >= 1) or overlap >= 2:
            return {"key": key, "matched_alias": alias, "score": min(score, 0.96), "method": "fuzzy + ontology match"}
        return None

    def search(self, query, k=4, preferred_category=None):
        query = normalize(query)
        q = self.vectorizer.transform([query])
        cosine = cosine_similarity(q, self.matrix)[0]
        q_tokens = self._tokens(query)
        ranked = []
        for i, doc in enumerate(self.docs):
            item_tokens = self._tokens(doc["item"])
            keyword_tokens = self._tokens(doc["keywords"])
            overlap = len(q_tokens & (item_tokens | keyword_tokens))
            exact_phrase = doc["item"].lower() in query or query in doc["item"].lower()
            category_bonus = 0.18 if preferred_category and doc.get("category_key") == preferred_category else 0.0
            score = float(cosine[i]) + min(overlap / max(len(q_tokens), 1), 1.0) * 0.30 + category_bonus
            if exact_phrase:
                score += 0.65
            ranked.append((score, float(cosine[i]), overlap, exact_phrase, i))
        ranked.sort(reverse=True)

        results = []
        for score, cosine_score, overlap, exact_phrase, i in ranked[:max(k, 8)]:
            if not exact_phrase and cosine_score < 0.06 and overlap == 0:
                continue
            results.append({**self.docs[i], "score": min(score, 1.0), "cosine_score": cosine_score, "overlap": overlap, "exact_match": exact_phrase})
            if len(results) >= k:
                break
        return results

    @staticmethod
    def confidence(result, generic=None):
        if generic and generic.get("score", 0) >= 0.93:
            return 98
        if result.get("exact_match"):
            return 97
        score = result.get("score", 0.0)
        return max(1, min(94, round(score * 100)))
