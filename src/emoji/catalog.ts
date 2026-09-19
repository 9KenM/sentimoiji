import emojibase from "emojibase-data/en/data.json" with { type: "json" };
import meta from "emojibase-data/meta/groups.json" with { type: "json" };

export interface Emoji {
  character: string;
  name: string;
  subgroup: string;
}

const groupNames: Record<string, string> = meta.groups;
const subgroupNames: Record<string, string> = meta.subgroups;

const isStandaloneEmoji = (entry: { group?: number; subgroup?: number }) =>
  entry.group !== undefined &&
  entry.subgroup !== undefined &&
  groupNames[entry.group] !== "component";

export const emojiCatalog: Emoji[] = emojibase
  .filter(isStandaloneEmoji)
  .map((entry) => ({
    character: entry.emoji,
    name: entry.label,
    subgroup: subgroupNames[entry.subgroup!],
  }));
