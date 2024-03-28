from FlagEmbedding import BGEM3FlagModel

class InferlessPythonModel:
    def initialize(self):
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def infer(self, inputs):
        sentences = inputs["sentences"]
        embeddings = self.model.encode(sentences, 
                            batch_size=12, 
                            max_length=8192)['dense_vecs']
        return {"embeddings": list(embeddings)}
    def finalize(self, args):
        self.pipe = None
