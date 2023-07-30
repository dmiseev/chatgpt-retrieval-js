import 'dotenv/config';
import readline from 'readline';
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { NotionLoader } from "langchain/document_loaders/fs/notion";

// Entry point of the application
async function main() {
    const chain = await setupLLMChain();
    await endlessLoop(chain);
    console.log('End of the loop.');
}

// Load documents to LLM and create a retrieval chain
async function setupLLMChain() {
    console.log("Loading documents...");
    const plainDocs = await loadPlainDocuments();
    const markdownDocs = await loadMarkdownDocuments();
    const docs = [...plainDocs, ...markdownDocs];
    console.log("Documents loaded.");

    console.log("Splitting documents...");
    const splitDocs = await splitDocuments(docs);
    console.log("Documents split.");

    console.log("Storing documents to memory vector store...");
    const vectorStore = await storeDocuments(splitDocs);
    console.log("Documents stored to memory vector store.");

    console.log("Creating retrieval chain...");
    const model = new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        openAIApiKey: process.env.OPENAI_API_KEY,
        temperature: 0,
    });

    return RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
}

// Load documents from the specified directory
async function loadPlainDocuments() {
    const loader = new DirectoryLoader("./data", {
        ".json": (path) => new JSONLoader(path),
        ".txt": (path) => new TextLoader(path),
        ".csv": (path) => new CSVLoader(path),
    });

    return loader.load();
}

// Load markdown documents from the specified directory
async function loadMarkdownDocuments() {
    const directoryPath = "./data";
    const loader = new NotionLoader(directoryPath);

    return await loader.load();
}

// Split documents using the text splitter
async function splitDocuments(docs) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 0,
    });

    return textSplitter.splitDocuments(docs);
}

// Store split documents in the memory vector store
async function storeDocuments(splitDocs) {
    return MemoryVectorStore.fromDocuments(splitDocs, new OpenAIEmbeddings());
}

// Endless loop to take user input and get responses from GPT
async function endlessLoop(chain) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    while (true) {
        try {
            const query = await new Promise((resolve) => {
                rl.question('\x1b[33mEnter your question (or type "exit" to quit): \x1b[0m', (answer) => {
                    resolve(answer.trim());
                });
            });

            if (query.toLowerCase() === 'exit') {
                console.log('\x1b[32mExiting the loop...\x1b[0m');
                rl.close();
                break;
            }

            console.log('\x1b[36mYou entered:\x1b[0m', `${query}`);

            const response = await chain.call({
                query: query
            });

            console.log('\x1b[32mAI answered:\x1b[0m', `${response.text}`);
        } catch (error) {
            console.error('An error occurred:', error);
        }
    }
}

// Start the application
await main();
