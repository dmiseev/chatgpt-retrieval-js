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
import {OpenAI} from "langchain/llms/openai";

// Constants
const DATA_DIR = "./data";
// const OPENAI_API_MODEL_NAME = 'gpt-3.5-turbo';
const OPENAI_API_MODEL_NAME = 'gpt-4';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Entry point of the application
async function main() {
    const chain = await setupLLMChain();
    const response = await chain.call({
        query: 'Act as a PHP developer. You have to write changelogs which describe your changes based on diff files. \n' +
            'Facade class pattern is "*Facade.php".\n' +
            'Transfer object definitions are stored in "*.transfer.xml" files.There are several type of changelogs:\n\n\n' +
            '1a. You added a new method to facade class. Expected output is the following: "* Introduced \`FACADE::METHOD()\` facade method.\" Where FACADE is affected class name and METHOD is affected method name.\n' +
            '1b. You removed an existing method from facade class. Expected output is the following: "* Removed \`FACADE::METHOD() facade method.\` Where FACADE is affected class name and METHOD is affected method name.\n' +
            '2a. You added a new transfer object definition. Expected output is the following: "* Introduced \`TRANSFER\` transfer object." where TRANSFER is the name of the object from xml file. Ignore its properties for such case.\n' +
            '2b. You deleted an existing transfer object definition. Expected output is the following: "* Removed \`TRANSFER\` transfer object." where TRANSFER is the name of the object from xml file. Ignore its properties for such case.\n' +
            '2c. You added a new transfer object property to existing transfer object definition. Transfer object definitions are stored in ".transfer.xml" files. Expected output is the following: "* Introduced \`TRANSFER.PROPERTY\` transfer property." where TRANSFER is the name of the object and PROPERTY is a name of the property inside of object.\n' +
            '2d. You deleted an existing transfer object property from existing transfer object definition. Transfer object definitions are stored in ".transfer.xml" files. Expected output is the following: "* Removed \`TRANSFER.PROPERTY\` transfer property." where TRANSFER is the name of the object and PROPERTY is a name of the property inside of object.\n' +
            '3. In case a new transfer object definition is added you do not need to describe added properties, only added transfer object is mentioned.\n' +
            '4. "*FacadeInterface.php" files should be skipped.\n\n' +
            'Only diffs starting from "+" and "-" are relevant for you.\n' +
            'You need to list all affected transfers and facade methods according to diffs.\n' +
            'Combine all changelogs from all documents in the single snippet.'
    });

    console.log({response});
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
    console.log(`Documents split to ${splitDocs.length} docs.`);

    console.log("Storing documents to memory vector store...");
    const vectorStore = await storeDocuments(splitDocs);
    console.log("Documents stored to memory vector store.");

    console.log("Creating retrieval chain...");
    const model = new OpenAI({
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
        chunkSize: 8000, // 8000
        chunkOverlap: 0,
    });

    return textSplitter.splitDocuments(docs);
}

// Store split documents in the memory vector store
async function storeDocuments(splitDocs) {
    return MemoryVectorStore.fromDocuments(splitDocs, new OpenAIEmbeddings());
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
