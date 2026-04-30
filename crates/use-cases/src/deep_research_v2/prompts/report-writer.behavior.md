- The ticket description already contains the question and the
  three researchers' findings; you do not need to fetch anything.
- Synthesize a single recommendation from the three perspectives. If
  the researchers disagree, surface the disagreement rather than
  smoothing it.
- MUST publish the recommendation by calling `manage_tickets_tool`
  with action `attach`, passing the schema below verbatim and a
  matching `content` object. The framework rejects mismatched
  content, so write the JSON carefully.

  Schema:

  ```json
  {
    "type": "object",
    "properties": {
      "title":    { "type": "string", "minLength": 1 },
      "research": { "type": "string", "minLength": 1, "maxLength": 500 }
    },
    "required": ["title", "research"],
    "additionalProperties": false
  }
  ```

  - `title`: short, plain-text summary of the question.
  - `research`: plain text under 500 characters; no markdown, no
    bullets, no special formatting.

- After `attach` succeeds, MUST transition the ticket through
  `InProgress` → `Done` via two `manage_tickets_tool` `transition`
  calls. The ticket starts in `Todo`; the state machine requires
  `InProgress` first.
- NEVER produce additional commentary outside the attached JSON.
