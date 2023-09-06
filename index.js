import 'dotenv/config';
import readline from 'readline';
import fs from "fs";
import path from "path";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { DirectoryLoader, UnknownHandling } from "langchain/document_loaders/fs/directory";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { NotionLoader } from "langchain/document_loaders/fs/notion";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

// Constants
const DATA_DIR = "./data";
// const OPENAI_API_MODEL_NAME = 'gpt-3.5-turbo';
const OPENAI_API_MODEL_NAME = 'gpt-4';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

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
    const pdfDocs = await loadPdfDocuments();
    const docs = [...plainDocs, ...markdownDocs, ...pdfDocs];
    console.log("Documents loaded.");

    console.log("Splitting documents...");
    const splitDocs = await splitDocuments(docs);
    console.log("Documents split.");

    console.log("Storing documents to memory vector store...");
    const vectorStore = await storeDocuments(splitDocs);
    console.log("Documents stored to memory vector store.");

    console.log("Creating retrieval chain...");
    const model = new ChatOpenAI({
        modelName: OPENAI_API_MODEL_NAME,
        openAIApiKey: OPENAI_API_KEY,
        temperature: 0,
    });

    return RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
        returnSourceDocuments: true,
    });
}

// Load documents from the specified directory
async function loadPlainDocuments() {
    const loader = new DirectoryLoader(DATA_DIR, {
        ".json": (path) => new JSONLoader(path),
        ".txt": (path) => new TextLoader(path),
        ".csv": (path) => new CSVLoader(path),
    }, true, UnknownHandling.Ignore);

    return loader.load();
}

// Load markdown documents from the specified directory
async function loadMarkdownDocuments() {
    const loader = new NotionLoader(DATA_DIR);

    return await loader.load();
}

// Load PDF documents from the specified directory
async function loadPdfDocuments() {
    let pdfs = [];
    const fileNames = scanForPDFs(DATA_DIR);

    for (const fileName of fileNames) {
        const loader = new PDFLoader(`${fileName}`);
        const pdfDocs = await loader.load();
        pdfs = [...pdfs, ...pdfDocs];
    }

    return pdfs;
}

// Split documents using the text splitter
async function splitDocuments(docs) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 8000,
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
            console.log('\x1b[32mSource Document:\x1b[0m', `${response.sourceDocuments[0].metadata.source}`);
        } catch (error) {
            console.error('An error occurred:', error);
        }
    }
}

// Function to recursively scan a directory for PDF files
function scanForPDFs(directory) {
    const pdfFiles = [];

    function scanDir(dir) {
        const files = fs.readdirSync(dir);

        for (const file of files) {
            const filePath = path.join(dir, file);
            const stats = fs.statSync(filePath);

            if (stats.isFile() && path.extname(file).toLowerCase() === '.pdf') {
                pdfFiles.push(filePath);
            } else if (stats.isDirectory()) {
                scanDir(filePath);
            }
        }
    }

    scanDir(directory);
    return pdfFiles;
}

// Start the application
await main();
