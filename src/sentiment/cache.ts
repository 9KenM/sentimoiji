export function withCache<T>(
  analyze: (text: string) => Promise<T>,
  maxEntries: number,
) {
  const results = new Map<string, Promise<T>>();

  return (text: string): Promise<T> => {
    const cached = results.get(text);
    if (cached) return cached;

    const result = analyze(text);
    results.set(text, result);
    result.catch(() => results.delete(text));
    if (results.size > maxEntries) {
      results.delete(results.keys().next().value!);
    }
    return result;
  };
}
