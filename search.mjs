import { readFile } from "node:fs/promises";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { VectorStore } from "@langchain/core/vectorstores";

class MemoryVectorStore extends VectorStore {
  constructor(embeddings) {
    super(embeddings, {});
    this.embeddings = embeddings;
    this.documents = [];
    this.vectors = [];
  }

  _vectorstoreType() {
    return "memory";
  }

  async addDocuments(documents) {
    const texts = documents.map((doc) => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    return this.addVectors(vectors, documents);
  }

  async addVectors(vectors, documents, options) {
    const ids = [];
    for (let i = 0; i < documents.length; i++) {
      this.documents.push(documents[i]);
      this.vectors.push(vectors[i]);
      ids.push((this.documents.length - 1).toString());
    }
    return ids;
  }

  async similaritySearchVectorWithScore(query, k, filter) {
    const scores = this.vectors.map((vector, idx) => {
      const dotProduct = vector.reduce(
        (sum, val, i) => sum + val * query[i],
        0
      );
      const magnitudeA = Math.sqrt(
        vector.reduce((sum, val) => sum + val * val, 0)
      );
      const magnitudeB = Math.sqrt(
        query.reduce((sum, val) => sum + val * val, 0)
      );
      const similarity = dotProduct / (magnitudeA * magnitudeB);
      return { idx, score: similarity };
    });

    scores.sort((a, b) => b.score - a.score);
    return scores
      .slice(0, k)
      .map(({ idx, score }) => [this.documents[idx], score]);
  }

  async delete(params) {
    // Simple implementation - clear all if no params, otherwise would need filtering logic
    if (!params || Object.keys(params).length === 0) {
      this.documents = [];
      this.vectors = [];
    }
    // For this simple implementation, we'll just clear all
    // In a real implementation, you'd filter based on params
  }
}

const products = JSON.parse(await readFile("products.json", "utf-8"));

function createStore(products) {
  const embeddings = new OpenAIEmbeddings();
  const docs = products.map(
    (product) =>
      new Document({
        pageContent: `${product.name} - ${product.description} - ${product.price}`,
        metadata: {
          source: product.id,
        },
      })
  );
  const vectorStore = new MemoryVectorStore(embeddings);
  return vectorStore.addDocuments(docs).then(() => vectorStore);
}

const store = await createStore(products);

async function search(query, count = 3) {
  const results = await store.similaritySearch(query, count);
  return results.map((result) => ({
    ...result.metadata,
    content: result.pageContent,
  }));
}

const results = await search("I want to buy a watch");
console.log(results);
