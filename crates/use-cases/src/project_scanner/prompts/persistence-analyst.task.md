Audit {file} for persistence and backdoor issues: code that creates a foothold surviving restart, hides ongoing remote access, or quietly bypasses auth.

In scope: install or postinstall hooks writing to ~/.ssh/authorized_keys, ~/.bashrc, cron, systemd, launchd; binary or PATH replacement; hidden HTTP routes that grant access; magic-token or env-var auth shortcuts; outbound callbacks from startup or install paths to attacker-controlled hosts.

Out of scope: documented admin tooling behind real auth, dev-only paths clearly gated by NODE_ENV or DEBUG, and standard package-manager native-module postinstalls. Skip silently.

Output (via tool calls):
- For each (source -> path -> sink) you can trace, call report_issue once with the sink line, a severity (low, medium, high), a short category slug (e.g. 'auth_bypass', 'startup_hook'), and a trace formatted 'L<src> <what>  ->  L<mid> <what>  ->  L<sink> <what>'. If a leg is not visible in this file, write '?? (not visible in this file)' for that leg rather than dropping the finding.
- Call mark_status exactly once as the final action: status 'complete' / 'partial' / 'blocked', trustworthy true only if the file is free of persistence concerns, and a one-line summary at most 200 chars stating the conclusion.

MUST anchor every issue to a line number from this file. NEVER report findings outside the persistence and backdoor scope. Budget: at most {budget} tool calls total.
