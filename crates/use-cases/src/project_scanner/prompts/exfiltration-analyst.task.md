Audit `{file}` for data-exfiltration issues: code that ships sensitive material off the host by carrying a credential, secret, key, cookie, or PII from a source in this program to a sink that leaves the machine.

### Part A: Scope

In scope: env-var or file reads of credentials, tokens, signing keys, `.env`, `~/.ssh/*`, `~/.aws/*`, keychain, browser cookie or login databases; PII (email, phone, SSN-shaped strings); crash or panic hooks capturing in-memory secrets. Sinks include HTTP(S) requests to non-localhost hosts, DNS labels carrying data, WebSocket / gRPC / raw socket writes to remote endpoints, telemetry or analytics SDKs with overbroad payloads, log lines emitting secrets when stdout is shipped, URL query or path segments built from secrets, and crash-dump or error-reporter uploads.

Out of scope: requests to localhost, 127.0.0.1, or unix sockets; files the calling code clearly owns and was explicitly asked to upload; test fixtures and mocked HTTP clients. Skip silently.

### Part B: Output requirements

MUST call `report_issue` once per traced (source -> path -> sink) flow, with the sink line, severity (low / medium / high), a short category slug (e.g. `env_exfil`, `cookie_exfil`), and a trace formatted `L<src> <what>  ->  L<mid> <what>  ->  L<sink> <what>`. If a leg is not visible in this file, write `?? (not visible in this file)` for that leg rather than dropping the finding.

MUST call `mark_status` exactly once as the final action: status `complete` / `partial` / `blocked`, `trustworthy` true only if the file is free of exfiltration concerns, and a one-line summary at most 200 chars stating the conclusion.

MUST anchor every issue to a line number from this file.
NEVER report findings outside the data-exfiltration scope.
IMPORTANT: budget at most {budget} tool calls total.
