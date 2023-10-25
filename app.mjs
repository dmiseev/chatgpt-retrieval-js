import express from 'express';
import fs from 'fs';
import path from 'path';
import {main} from './index.js';
import http from 'http';
import {WebSocketServer} from 'ws';
import MarkdownIt from 'markdown-it';

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });
const port = 3000;
const md = new MarkdownIt();

const clients = [];
wss.on('connection', (ws) => {
    clients.push(ws);

    ws.on('close', () => {
        // Remove the WebSocket connection when a client disconnects
        const index = clients.indexOf(ws);
        if (index !== -1) {
            clients.splice(index, 1);
        }
    });
});

server.listen(3001, () => {
    console.log('Server is running on port 3001');
});

// Initialize an empty array to store chat history
const chatHistory = [];
let LLMChain = null;

app.set('view engine', 'ejs');
app.use(express.static('public'));

// Define a route to list files in a directory
app.get('/listFiles', (req, res) => {
    const folderPath = './data'; // Change this to the path of the folder you want to list
    const files = fs.readdirSync(folderPath);
    const fileDetails = files.map((file) => {
        const filePath = path.join(folderPath, file);
        const fileStat = fs.statSync(filePath);
        return {
            name: file,
            extension: path.extname(file),
            creationDate: fileStat.birthtime,
        };
    });
    res.json(fileDetails);
});

app.get('/', (req, res) => {
    res.render('index');
});

app.listen(port, async () => {
    console.log(`Server is running on http://localhost:${port}`);
    LLMChain = await main();
    sendMessage("AI: All documents loaded successfully.");
});

// Define a route to send and receive chat messages
app.get('/chat', (req, res) => {
    res.json(chatHistory);
});

app.post('/chat', express.json(), async (req, res) => {
    if (req.body.message) {
        sendMessage("You: " + req.body.message);
        res.status(200).send('Message received and broadcasted.');

        const response = await LLMChain.call({
            query: req.body.message
        });

        const renderedHTML = md.render(response.text);

        sendMessage(`AI: ${renderedHTML} <p>Source: ${extractFinalPath(response.sourceDocuments[0].metadata.source)}</p>`);
    } else {
        res.status(400).send('Message is required.');
    }
});

function sendMessage(message) {
    chatHistory.push(message);

    // Broadcast the message to all connected clients
    clients.forEach((client) => {
        if (client.readyState === 1) {
            client.send(message);
        }
    });
}

function extractFinalPath(inputPath) {
    const dataIndex = inputPath.indexOf('/data/');

    if (dataIndex !== -1) {
        return inputPath.slice(dataIndex + '/data/'.length);
    } else {
        // Handle the case when "/data/" is not found in the path
        return inputPath;
    }
}