import json
import os
import requests


class GraniteClient:
    """IBM watsonx.ai / Granite integration for grounded classification and generation."""

    def __init__(self):
        self.api_key = os.getenv("WATSONX_APIKEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.region = os.getenv("WATSONX_REGION", "us-south")
        self.model_id = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")
        self.timeout = int(os.getenv("WATSONX_TIMEOUT", "45"))

    @property
    def enabled(self):
        return bool(self.api_key and self.project_id)

    def _base(self):
        return {
            "us-south": "https://us-south.ml.cloud.ibm.com",
            "eu-de": "https://eu-de.ml.cloud.ibm.com",
            "eu-gb": "https://eu-gb.ml.cloud.ibm.com",
            "jp-tok": "https://jp-tok.ml.cloud.ibm.com",
            "au-syd": "https://au-syd.ml.cloud.ibm.com",
            "ca-tor": "https://ca-tor.ml.cloud.ibm.com",
            "ap-south-1": "https://ap-south-1.aws.wxai.ibm.com",
        }.get(self.region, "https://us-south.ml.cloud.ibm.com")

    def _token(self):
        r = requests.post(
            "https://iam.cloud.ibm.com/oidc/token",
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["access_token"]

    def chat(self, system, user, max_tokens=500):
        if not self.enabled:
            return None
        token = self._token()
        url = f"{self._base()}/ml/v1/text/chat?version=2025-10-25"
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "project_id": self.project_id,
            "model_id": self.model_id,
            "max_completion_tokens": max_tokens,
            "temperature": 0,
        }
        r = requests.post(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def classify(self, item, categories):
        """Ask Granite for a category only after deterministic retrieval/classification."""
        system = (
            "You are the Shuddham.AI waste classifier. Choose exactly one category from the supplied list. "
            "Do not invent a category. Return ONLY compact JSON with keys category_key, confidence, rationale. "
            "Confidence must be an integer 0-100. If the input is not clearly a waste item, return category_key=unknown."
        )
        options = "\n".join(f"{k}: {v}" for k, v in categories.items())
        raw = self.chat(system, f"Categories:\n{options}\n\nUser item: {item}", max_tokens=180)
        if not raw:
            return None
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            return data
        except Exception:
            return None
