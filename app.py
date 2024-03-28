import json
from sentence_transformers import SentenceTransformer, util

class InferlessPythonModel:
    def initialize(self):
        self.model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en",trust_remote_code=True)
        # control your input sequence length up to 8192
        self.model.max_seq_length = 1024

    def infer(self, inputs):
        sentences = inputs["sentences"]
        embeddings = self.model.encode(sentences)
        return {"result": embeddings}
    def finalize(self, args):
        self.pipe = None
