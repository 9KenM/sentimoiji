import assert from "node:assert/strict";
import { test } from "node:test";
import { emojiCatalog } from "../emoji/catalog.ts";
import {
  MAX_OPTIONS_PER_CHOICE,
  buildQuestions,
  createSentimentAnalyzer,
} from "./analyzer.ts";
import type { DecisionModel } from "./analyzer.ts";

const SUBGROUP_QUESTION = "subgroup";

function modelFavouring(subgroup: string, character: string): DecisionModel {
  return {
    async systemOne({ questions }) {
      const answers = Object.fromEntries(
        Object.entries(questions).map(([name, question]) => {
          const isSubgroupQuestion = name === SUBGROUP_QUESTION;
          const favourite = isSubgroupQuestion ? subgroup : character;
          const favouriteProbability = isSubgroupQuestion ? 0.9 : 0.5;
          const probabilities = Object.fromEntries(
            Object.keys(question.criteria).map((option) => [
              option,
              option === favourite ? favouriteProbability : 0.0001,
            ]),
          );
          return [
            name,
            { type: "choice" as const, choice: favourite, confidence: 0.9, probabilities },
          ];
        }),
      );
      return { model: "fake", answers, usage: { input_tokens: 0, output_tokens: 0 } };
    },
  };
}

test("catalog holds the full emoji set without skin tone variants", () => {
  const characters = emojiCatalog.map(({ character }) => character);
  assert.ok(characters.length > 1900);
  assert.equal(new Set(characters).size, characters.length);
  for (const expected of ["😀", "❤️‍🔥", "👩‍🚀", "🏳️‍🌈", "🇯🇵", "#️⃣"]) {
    assert.ok(characters.includes(expected), `missing ${expected}`);
  }
  assert.ok(!characters.includes("👋🏽"));
});

test("every emoji is offered exactly once within the Choice option limit", () => {
  const { [SUBGROUP_QUESTION]: subgroupQuestion, ...emojiQuestions } =
    buildQuestions(emojiCatalog);
  const offered = Object.values(emojiQuestions).flatMap(({ criteria }) =>
    Object.keys(criteria),
  );

  assert.deepEqual(offered, emojiCatalog.map(({ character }) => character));
  for (const { criteria } of [subgroupQuestion, ...Object.values(emojiQuestions)]) {
    assert.ok(Object.keys(criteria).length <= MAX_OPTIONS_PER_CHOICE);
  }
});

test("picks the emoji favoured by both its subgroup and its own question", async () => {
  const rocket = emojiCatalog.find(({ character }) => character === "🚀")!;
  const analyze = createSentimentAnalyzer(
    modelFavouring(rocket.subgroup, rocket.character),
    emojiCatalog,
  );

  assert.deepEqual(await analyze("We have liftoff!"), rocket);
});

test("a forced winner in an unlikely subgroup loses to a likelier subgroup", async () => {
  const flag = emojiCatalog.find(({ character }) => character === "🇯🇵")!;
  const analyze = createSentimentAnalyzer(
    modelFavouring("face-smiling", flag.character),
    emojiCatalog,
  );

  assert.equal((await analyze("What a lovely day")).subgroup, "face-smiling");
});
