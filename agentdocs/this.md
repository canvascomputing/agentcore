# This

How every file under `agentdocs/` is written. This file is itself an example of the format.

## 1. File shape

**One topic per file. Start with a title and a one-sentence description.**

- `# Title`: one word or short phrase, no trailing punctuation.
- One sentence under the title that states what the file covers.
- Sections are numbered: `## 1. Title Cased Heading`.
- Each section is self-contained; a reader can skip to it directly.

## 2. Section shape

**Bold rule first. Bullets second. A closing sentence is optional.**

- The first line after the heading is a bold one-liner stating the rule.
- The rule is an instruction, not a description.
- Bullets follow, optionally preceded by a one-line framing sentence.
- A closing sentence is added only when it carries information the bullets do not.

## 3. Bullets

**Three to five bullets per section. One line each. Imperative voice.**

- Start with a capital letter; end with a period.
- Lead with the verb or with the thing being forbidden.
- Two short sentences per bullet are acceptable; longer bullets are not.
- Nested bullets are used only under a parent line ending in a colon.

## 4. Enumerations

**Use bullets, not tables.**

- Tables produce wide rows that are hard to compare.
- For `name: description` pairs, write `` `Name`: description. ``
- Group related bullets under a one-line header ending in a colon.
- Code fences are acceptable for commands and small code examples.

## 5. Punctuation

**Colons, not em dashes.**

- Use `:` where an em dash would otherwise appear.
- Use commas or parentheses for short parenthetical asides.
- `>` blockquotes are reserved for callouts at the top of a file.

## 6. Voice

**Direct and neutral. No marketing language. No unnecessary jargon.**

- State the rule; justify only when the rule is not obvious on its own.
- Prefer present tense and second person over passive voice.
- Avoid adjectives that do not carry information ("powerful", "clean", "seamless").
- Avoid borrowed metaphors ("kernel", "plane", "seam", "pipeline") unless they are the precise technical term.
