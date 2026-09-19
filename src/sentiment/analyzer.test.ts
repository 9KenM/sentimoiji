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

function modelFavouring(
  subgroup: string,
  emojiNames: string[],
  otherProbability = 0.0001,
): DecisionModel {
  return {
    async systemOne({ questions }) {
      const answers = Object.fromEntries(
        Object.entries(questions).map(([name, question]) => {
          const isSubgroupQuestion = name === SUBGROUP_QUESTION;
          const favourites = isSubgroupQuestion ? [subgroup] : emojiNames;
          const favouriteProbability = isSubgroupQuestion ? 0.9 : 0.5;
          const options = Object.keys(question.criteria);
          const probabilities = Object.fromEntries(
            options.map((option) => [
              option,
              favourites.includes(option) ? favouriteProbability : otherProbability,
            ]),
          );
          const selected = options.find((option) => favourites.includes(option));
          return [
            name,
            {
              type: "choice" as const,
              choice: selected ?? options[0],
              confidence: 0.9,
              probabilities,
            },
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

  assert.deepEqual(offered, emojiCatalog.map(({ name }) => name));
  for (const { criteria } of [subgroupQuestion, ...Object.values(emojiQuestions)]) {
    assert.ok(Object.keys(criteria).length <= MAX_OPTIONS_PER_CHOICE);
  }
});

test("picks the emoji favoured by both its subgroup and its own question", async () => {
  const rocket = emojiCatalog.find(({ character }) => character === "🚀")!;
  const analyze = createSentimentAnalyzer(
    modelFavouring(rocket.subgroup, [rocket.name]),
    emojiCatalog,
  );

  assert.deepEqual(await analyze("We have liftoff!"), rocket);
});

test("a forced winner in an unlikely subgroup loses to a likelier subgroup", async () => {
  const flag = emojiCatalog.find(({ character }) => character === "🇯🇵")!;
  const grin = emojiCatalog.find(({ character }) => character === "😀")!;
  const analyze = createSentimentAnalyzer(
    modelFavouring(grin.subgroup, [flag.name, grin.name]),
    emojiCatalog,
  );

  assert.deepEqual(await analyze("What a lovely day"), grin);
});

test("a confidently wrong subgroup cannot zero out the favoured emoji", async () => {
  const anguished = emojiCatalog.find(({ character }) => character === "😧")!;
  const analyze = createSentimentAnalyzer(
    modelFavouring("face-negative", [anguished.name], 0),
    emojiCatalog,
  );

  assert.deepEqual(await analyze("This is the worst day of my life"), anguished);
});
