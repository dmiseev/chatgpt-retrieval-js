import 'dotenv/config';
import { OpenAI } from "langchain/llms/openai";
import {loadQAMapReduceChain, loadSummarizationChain} from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

// In this example, we use a `MapReduceDocumentsChain` specifically prompted to summarize a set of documents.
const text = fs.readFileSync("data/example.txt", "utf8");
const model = new OpenAI({
    modelName: 'gpt-4',
    temperature: 0
});
const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
const docs = await textSplitter.createDocuments([text]);
console.log(docs.length);

// This convenience function creates a document chain prompted to summarize a set of documents.
const chain = loadQAMapReduceChain(model);
const res = await chain.call({
    input_documents: docs,
    question: 'Act as a PHP developer. You have to write changelogs which describe your changes based on diff files. \n' +
        'Facade class pattern is "*Facade.php".\n' +
        'Transfer object definitions are stored in "*.transfer.xml" files.There are several type of changelogs:\n\n\n' +
        '1a. You added a new method to facade class. Expected output is the following: "* Introduced \`FACADE::METHOD() facade method.\` Where FACADE is affected class name and METHOD is affected method name.\n' +
        '1b. You removed an existing method from facade class. Expected output is the following: "* Removed \`FACADE::METHOD() facade method.\` Where FACADE is affected class name and METHOD is affected method name.\n' +
        '2a. You added a new transfer object definition. Expected output is the following: "* Introduced \`TRANSFER\` transfer object." where TRANSFER is the name of the object from xml file. Ignore its properties for such case.\n' +
        '2b. You deleted an existing transfer object definition. Expected output is the following: "* Removed \`TRANSFER\` transfer object." where TRANSFER is the name of the object from xml file. Ignore its properties for such case.\n' +
        '2c. You added a new transfer object property to existing transfer object definition. Transfer object definitions are stored in ".transfer.xml" files. Expected output is the following: "* Introduced \`TRANSFER.PROPERTY\` transfer property." where TRANSFER is the name of the object and PROPERTY is a name of the property inside of object.\n' +
        '23. You deleted an existing transfer object property from existing transfer object definition. Transfer object definitions are stored in ".transfer.xml" files. Expected output is the following: "* Removed \`TRANSFER.PROPERTY\` transfer property." where TRANSFER is the name of the object and PROPERTY is a name of the property inside of object.\n' +
        '4. In case a new transfer object definition is added you do not need to describe added properties, only added transfer object is mentioned.\n' +
        '5. "*FacadeInterface.php" files should be skipped.\n\n' +
        'Only diffs starting from "+" and "-" are relevant for you.\n' +
        'You need to list all affected transfers and facade methods according to diffs.\n' +
        'Combine all changelogs from all split docs in the single snippet.'
});
console.log({ res });