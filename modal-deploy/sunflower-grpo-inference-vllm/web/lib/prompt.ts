export function buildTranslationPrompt({
  source,
  target,
  text,
}: {
  source: string;
  target: string;
  text: string;
}) {
  return `translate from ${source.toLowerCase()} to ${target.toLowerCase()}: ${text.trim()}`;
}
