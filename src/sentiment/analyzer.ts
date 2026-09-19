import { choice } from "@typesafe-ai/sdk";
import type {
  ChoiceQuestion,
  SystemOneRequest,
  SystemOneResult,
} from "@typesafe-ai/sdk";
import type { Emoji } from "../emoji/catalog.ts";

type ChoiceQuestions = Record<string, ChoiceQuestion>;

export interface DecisionModel {
  systemOne(
    request: SystemOneRequest<ChoiceQuestions>,
  ): PromiseLike<SystemOneResult<ChoiceQuestions>>;
}

export const MAX_OPTIONS_PER_CHOICE = 255;

const SUBGROUP_QUESTION = "subgroup";
const SUBGROUP_INSTRUCTIONS =
  "Which category contains the emoji that best expresses the sentiment and emotion of the text?";
const EMOJI_INSTRUCTIONS =
  "Which of these emoji best expresses the sentiment and emotion of the text?";

const emojiQuestionName = (index: number) => `emoji_${index}`;

function chunk<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let start = 0; start < items.length; start += size) {
    chunks.push(items.slice(start, start + size));
  }
  return chunks;
}

export function buildQuestions(catalog: Emoji[]): ChoiceQuestions {
  const charactersBySubgroup: Record<string, string[]> = {};
  for (const { subgroup, character } of catalog) {
    (charactersBySubgroup[subgroup] ??= []).push(character);
  }

  const emojiQuestions = chunk(catalog, MAX_OPTIONS_PER_CHOICE).map(
    (emojis, index) => [
      emojiQuestionName(index),
      choice(
        EMOJI_INSTRUCTIONS,
        Object.fromEntries(emojis.map(({ character, name }) => [character, name])),
      ),
    ],
  );

  return {
    [SUBGROUP_QUESTION]: choice(SUBGROUP_INSTRUCTIONS, charactersBySubgroup),
    ...Object.fromEntries(emojiQuestions),
  };
}

export function pickEmoji(
  catalog: Emoji[],
  answers: SystemOneResult<ChoiceQuestions>["answers"],
): Emoji {
  const subgroupProbabilities = answers[SUBGROUP_QUESTION].probabilities;
  const likelihoods = catalog.map((emoji, index) => {
    const questionName = emojiQuestionName(
      Math.floor(index / MAX_OPTIONS_PER_CHOICE),
    );
    return (
      subgroupProbabilities[emoji.subgroup] *
      answers[questionName].probabilities[emoji.character]
    );
  });
  return catalog[likelihoods.indexOf(Math.max(...likelihoods))];
}

export function createSentimentAnalyzer(model: DecisionModel, catalog: Emoji[]) {
  const questions = buildQuestions(catalog);
  return async (text: string): Promise<Emoji> => {
    const { answers } = await model.systemOne({ state: text, questions });
    return pickEmoji(catalog, answers);
  };
}
