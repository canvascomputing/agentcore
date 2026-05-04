## Task

You are about to audit this project's `{extension}` files for code that fits one or more of the risk domains listed in your role description. Your inputs:

- technology: `{technology}`
- domains in scope: see your role description (each domain has a name and a one-line description there).

### Part A: Reframe each domain in `{technology}`'s vocabulary

Before you grep anything, take a moment for each domain in scope and think like a `{technology}` engineer who knows the language's standard library, ecosystem, and idioms. Indicators are **call sites of specific APIs and idiomatic constructions** — not English nouns, not security buzzwords.

For example, exfiltration in Python is essentially never the literal string `exfil` or `sensitive`; those words don't appear in real code. It looks like `requests.post(...)`, `urllib.request.urlopen(...)`, `socket.send(...)`, `subprocess.run(["curl", ...])`, `base64.b64encode(...)` paired with a network sink, calls to cloud SDKs (`boto3.client('s3').put_object(...)`, `gcs.upload_blob(...)`), or DNS lookups with constructed hostnames. Persistence in Python looks like writes to `~/.bashrc`, `~/.config/autostart/`, `crontab`, `subprocess.run(["systemctl", "enable", ...])`, or scheduled jobs via `schedule`/`apscheduler`. Translate every domain into `{technology}` the same way before you run a single search.

If `{extension}` is a non-executable format (Markdown, plain text, JSON, lockfiles), most domains will have no realistic indicators at all. That is a legitimate, quick outcome — settle with the "no indicators found" summary and stop.

### Part B: Use `grep_tool` correctly

IMPORTANT: `grep_tool` is **substring-only**. It does not understand regex, alternation, or character classes. A query like `base64|requests.post` matches the literal text `base64|requests.post` and returns nothing. So:

- MUST run one focused `grep_tool` call per indicator. Five small searches each returning a handful of real hits beat one broad search that returns the whole codebase.
- MUST include the `glob` parameter on every call to scope to `*{extension}`. NEVER let the search range over unrelated file types.
- Each search string MUST be the shortest substring that uniquely identifies the call site you're hunting for (`requests.post`, `b64encode`, `subprocess.run`, `crontab`, `os.system`).
- NEVER concatenate alternatives with `|`, `,`, or anything else. Run separate calls.
- Use `output_mode: "content"` with `context_lines: 2` when you need to see the surrounding code; otherwise leave it at the default.

### Part C: Think creatively, like someone trying to hide

The textbook indicators in Part A catch the easy cases. Risky code worth flagging is often *deliberately disguised* — written by someone who doesn't want a keyword scanner to find it. After your first pass through the obvious idioms, spend a handful of searches probing the disguises. The angles below are prompts for *your* thinking, not a checklist to run blindly. Pick the ones that fit `{technology}` and what you have already seen in the project, and invent variants the list does not name.

- **Aliasing.** Imports renamed to drop the giveaway: `import requests as r`, `from base64 import b64encode as enc`, `from urllib.request import urlopen as fetch`. Search for the canonical name AND a few likely aliases.
- **Dynamic dispatch.** Sensitive APIs called by string instead of attribute: `getattr(socket, "send")`, `__import__("urllib.request")`, `globals()["b64encode"]`, `eval(...)`, `exec(...)`. The substrings `__import__`, `getattr(`, `eval(`, and `exec(` are themselves worth searching for in any source-code-bearing language.
- **String construction to evade scanners.** URLs and payloads built piece-by-piece (`"htt" + "ps://..."`, `chr(104) + chr(116) + ...`), hex literals (`\x68\x74\x74`), or unusually long base64-shaped runs (40+ characters of `[A-Za-z0-9+/=]`) embedded directly in source.
- **Off-the-canonical-path channels.** Beyond the standard HTTP libraries: paste sites (`pastebin.com`, `transfer.sh`, `0x0.st`, `hastebin`), webhook endpoints (`discord.com/api/webhooks`, `hooks.slack.com`, `webhook.site`), DNS-tunnelling shapes, mail (`smtplib`, `sendmail`), SFTP (`paramiko`), git remotes pointed at unusual hosts.
- **Off-the-canonical-path persistence.** Beyond cron and shell rc files: `PYTHONSTARTUP`, `sitecustomize.py`, `setup.py` entry-point hijacks, `~/.config/`, `/etc/profile.d/`, `~/.ssh/authorized_keys`, scheduling via `at`, `systemd-run`, `launchctl`, `schtasks`, modifications to `__init__.py` files in vendored deps.
- **Suspicious credential reads.** Environment variables that look credential-shaped (`AWS_*`, `GITHUB_TOKEN`, `*_API_KEY`, `*_SECRET`), credential files (`~/.aws/credentials`, `~/.git-credentials`, `~/.npmrc`, `~/.docker/config.json`), browser cookie/keychain stores.
- **Staging in odd places.** Writes into `/tmp`, `/dev/shm`, `/var/tmp`, `%TEMP%`, hidden dotfiles dropped next to source, files with executable bits set in unusual locations.
- **Anti-analysis tells.** Time-based gates (`time.sleep`, `if datetime.now() > ...`), environment checks for sandboxes/CI (`os.environ.get("CI")`), virtual-machine fingerprinting, debugger detection.

Three creative searches that catch real disguised code beat ten textbook searches that catch only beginners.

### Part D: Filter hits before forwarding

A finding is a **real call site or code expression** in `{technology}` that fits one of the domains in scope. It is not:

- A hit inside a comment, docstring, or string literal that merely mentions the API.
- A README example, a test fixture, or vendored example code.
- A keyword match in unrelated code (a variable called `cache` is not by itself a persistence indicator; a function called `download_file` is not by itself an exfiltration indicator).

When `grep_tool` produces a hit, MUST `read_file` to see the surrounding code before deciding to forward it. Forward only if the excerpt is, on its own, defensible evidence that the named domain is happening here.

### Part E: Output requirements

- MUST forward each finding via `manage_tickets_tool` `action: "create"` exactly once. The forwarded ticket's `task` body has four lines (`source:`, `technology:`, `domain:`, `excerpt:`) — the `domain:` value MUST be one of the domain names listed in your role description.
- MUST settle your own ticket via `manage_tickets_tool` `action: "done"` with a one-line summary `result`. **This call always happens, even if you forwarded zero findings.** A run that ends without it is silently lost.
- NEVER forward a finding without a verbatim excerpt. NEVER forward an excerpt without a path. NEVER forward without a domain tag. NEVER include verdict or remediation language anywhere — that is the investigator's job, not yours.
- When in doubt, fewer findings beat noisy ones. Two well-grounded forwards or zero is a better outcome than twenty keyword false positives.
