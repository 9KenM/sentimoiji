import assert from "node:assert/strict";
import { test } from "node:test";
import { withCache } from "./cache.ts";

function countingAnalyzer() {
  const calls: string[] = [];
  const analyze = async (text: string) => {
    calls.push(text);
    if (text === "fail") throw new Error("unavailable");
    return text.toUpperCase();
  };
  return { calls, analyze };
}

test("repeated text is analyzed once", async () => {
  const { calls, analyze } = countingAnalyzer();
  const cached = withCache(analyze, 10);

  assert.equal(await cached("hello"), "HELLO");
  assert.equal(await cached("hello"), "HELLO");
  assert.deepEqual(calls, ["hello"]);
});

test("failures are not cached", async () => {
  const { calls, analyze } = countingAnalyzer();
  const cached = withCache(analyze, 10);

  await assert.rejects(cached("fail"));
  await assert.rejects(cached("fail"));
  assert.deepEqual(calls, ["fail", "fail"]);
});

test("the oldest entry is evicted beyond the limit", async () => {
  const { calls, analyze } = countingAnalyzer();
  const cached = withCache(analyze, 2);

  for (const text of ["a", "b", "c", "a"]) await cached(text);
  assert.deepEqual(calls, ["a", "b", "c", "a"]);
});
