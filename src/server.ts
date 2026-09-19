import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import { text } from "node:stream/consumers";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import { emojiCatalog } from "./emoji/catalog.ts";
import { createSentimentAnalyzer } from "./sentiment/analyzer.ts";
import { withCache } from "./sentiment/cache.ts";

const PORT = Number(process.env.PORT ?? 3000);
const MAX_CACHED_ANALYSES = 1000;

const STATIC_FILES: Record<string, { file: string; contentType: string }> = {
  "/": { file: "index.html", contentType: "text/html" },
  "/style.css": { file: "style.css", contentType: "text/css" },
};

const analyzeSentiment = withCache(
  createSentimentAnalyzer(new TypeSafeClient(), emojiCatalog),
  MAX_CACHED_ANALYSES,
);

function sendJson(response: ServerResponse, status: number, body: unknown) {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(body));
}

async function handleSentiment(request: IncomingMessage, response: ServerResponse) {
  const body = JSON.parse(await text(request));
  if (typeof body?.text !== "string" || !body.text.trim()) {
    return sendJson(response, 400, { error: 'Missing "text" parameter' });
  }
  const { character, name } = await analyzeSentiment(body.text.trim());
  sendJson(response, 200, { emoji: character, name });
}

async function handleStatic(request: IncomingMessage, response: ServerResponse) {
  const asset = STATIC_FILES[request.url ?? ""];
  if (request.method !== "GET" || !asset) {
    response.writeHead(404).end();
    return;
  }
  const content = await readFile(new URL(`../static/${asset.file}`, import.meta.url));
  response.writeHead(200, { "Content-Type": `${asset.contentType}; charset=utf-8` });
  response.end(content);
}

createServer((request, response) => {
  const handle =
    request.method === "POST" && request.url === "/sentiment"
      ? handleSentiment
      : handleStatic;
  handle(request, response).catch((error: Error) => {
    console.error(error);
    sendJson(response, error instanceof SyntaxError ? 400 : 502, { error: error.message });
  });
}).listen(PORT, () => console.log(`sentimoji listening on http://localhost:${PORT}`));
