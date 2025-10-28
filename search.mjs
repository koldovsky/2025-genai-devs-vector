import { readFile } from "node:fs/promises";
import { createInterface } from "node:readline";
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

// Initialize store once at startup
console.log("Initializing vector store...");
const store = await createStore(products);
console.log(`Vector store initialized with ${products.length} products.\n`);

async function search(query, count = 3) {
  const queryEmbedding = await store.embeddings.embedQuery(query);
  const results = await store.similaritySearchVectorWithScore(
    queryEmbedding,
    count
  );
  // results is an array of [Document, score]
  return results.map(([doc, score]) => ({
    ...doc.metadata,
    content: doc.pageContent,
    score: score,
  }));
}

// Create readline interface for interactive queries
const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuery() {
  rl.question(
    "Enter your search query (or 'exit'/'quit' to stop): ",
    async (input) => {
      const query = input.trim();

      // Check for exit commands
      if (
        !query ||
        query.toLowerCase() === "exit" ||
        query.toLowerCase() === "quit"
      ) {
        console.log("Exiting...");
        rl.close();
        process.exit(0);
        return;
      }

      try {
        console.log(`\nSearching for: "${query}"\n`);
        const results = await search(query);

        if (results.length === 0) {
          console.log("No results found.\n");
        } else {
          results.forEach((result, index) => {
            console.log(`${index + 1}. Score: ${result.score.toFixed(4)}`);
            console.log(`   Content: ${result.content}`);
            console.log(`   Source ID: ${result.source}\n`);
          });
        }
      } catch (error) {
        console.error("Error during search:", error.message);
        console.log();
      }

      // Continue the loop
      askQuery();
    }
  );
}

// Handle Ctrl+C gracefully
rl.on("SIGINT", () => {
  console.log("\n\nExiting...");
  rl.close();
  process.exit(0);
});

// Start the interactive loop
askQuery();
