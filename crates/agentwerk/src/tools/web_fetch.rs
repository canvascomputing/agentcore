//! Fetches a URL and returns its extracted text. Gives an agent access to external documentation the prompt cannot enumerate up front.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{ToolContext, ToolLike, ToolResult};

const MAX_URL_LENGTH: usize = 2000;
const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;
const DEFAULT_MAX_LENGTH: usize = 100_000;
const FETCH_TIMEOUT_SECS: u64 = 60;
const MAX_REDIRECT_HOPS: usize = 10;

const DESCRIPTION: &str = "\
Fetches content from a specified URL and returns it as text.

- Takes a URL and an optional prompt describing what to extract.
- Fetches the URL content, converts HTML to readable plain text.
- Non-HTML content (JSON, plain text, markdown, etc.) is returned as-is.
- Response is truncated to max_length characters (default: 100,000).
- HTTP URLs are automatically upgraded to HTTPS.
- This tool is read-only and does not modify any files.
- IMPORTANT: Will fail for authenticated or private URLs (e.g. Nextcloud, GitLab, Confluence, Jira). Check if a specialized tool provides authenticated access first.
- For GitHub URLs, prefer using the gh CLI via bash instead (e.g., gh pr view, gh issue view, gh api).
- When a URL redirects to a different host, the tool will report the redirect URL instead of following it. \
  Make a new request with the redirect URL to fetch the content.";

/// Fetch a URL and return its content as text. Read-only. HTML is converted
/// to plain text; HTTP is upgraded to HTTPS; cross-host redirects are
/// surfaced rather than followed.
pub struct WebFetchTool;

impl ToolLike for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "prompt": {
                    "type": "string",
                    "description": "What to extract or focus on from the page content"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Max response length in characters (default: 100000)"
                }
            },
            "required": ["url"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn call<'a>(
        &'a self,
        input: Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let Some(url) = input["url"].as_str() else {
                return Ok(ToolResult::error("Missing required parameter: url"));
            };
            let prompt = input["prompt"].as_str().unwrap_or("");
            let max_length = input["max_length"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_MAX_LENGTH);

            let validated_url = match validate_url(url) {
                Ok(u) => u,
                Err(msg) => return Ok(ToolResult::error(msg)),
            };

            let text = match fetch_url(&validated_url).await {
                Ok(text) => text,
                Err(msg) => return Ok(ToolResult::error(msg)),
            };
            if let FetchedContent::Redirect {
                original_url,
                redirect_url,
                status,
            } = &text
            {
                let msg = format!(
                    "REDIRECT DETECTED: The URL redirects to a different host.\n\n\
                     Original URL: {original_url}\n\
                     Redirect URL: {redirect_url}\n\
                     Status: {status}\n\n\
                     To fetch the content, make a new web_fetch request with the redirect URL."
                );
                return Ok(ToolResult::success(msg));
            }
            let FetchedContent::Page {
                body,
                status,
                content_type,
                byte_count,
            } = text
            else {
                unreachable!()
            };

            let output = format_output(
                url,
                prompt,
                &body,
                status,
                &content_type,
                byte_count,
                max_length,
            );
            Ok(ToolResult::success(output))
        })
    }
}

// -- Fetching -----------------------------------------------------------------

enum FetchedContent {
    Page {
        body: String,
        status: u16,
        content_type: String,
        byte_count: usize,
    },
    Redirect {
        original_url: String,
        redirect_url: String,
        status: u16,
    },
}

async fn fetch_url(url: &str) -> std::result::Result<FetchedContent, String> {
    // Manual redirect handling prevents open-redirect exploitation across domains.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| e.to_string())?;

    let response = follow_safe_redirects(&client, url).await?;
    if let FollowResult::CrossDomain {
        original_url,
        redirect_url,
        status,
    } = response
    {
        return Ok(FetchedContent::Redirect {
            original_url,
            redirect_url,
            status,
        });
    }
    let FollowResult::Ok(response) = response else {
        unreachable!()
    };

    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read response: {e}"))?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(format!(
            "Response too large: {} bytes (max {MAX_RESPONSE_BYTES})",
            bytes.len()
        ));
    }

    let raw_text = String::from_utf8_lossy(&bytes);
    let body = if content_type.contains("text/html") {
        strip_html(&raw_text)
    } else {
        raw_text.into_owned()
    };

    Ok(FetchedContent::Page {
        body,
        status,
        content_type,
        byte_count: bytes.len(),
    })
}

fn format_output(
    url: &str,
    prompt: &str,
    body: &str,
    status: u16,
    content_type: &str,
    byte_count: usize,
    max_length: usize,
) -> String {
    let mut output = String::new();

    if !prompt.is_empty() {
        output.push_str(&format!("Prompt: {prompt}\n\n"));
    }
    output.push_str(&format!(
        "URL: {url}\nStatus: {status}\nContent-Type: {content_type}\nSize: {byte_count} bytes\n\n",
    ));

    let remaining = max_length.saturating_sub(output.len());
    if body.len() > remaining {
        output.push_str(&body[..remaining]);
        output.push_str("\n\n[Content truncated...]");
    } else {
        output.push_str(body);
    }

    output
}

// -- Redirect safety ----------------------------------------------------------

enum FollowResult {
    Ok(reqwest::Response),
    CrossDomain {
        original_url: String,
        redirect_url: String,
        status: u16,
    },
}

/// Follows same-host redirects (including www. add/remove) up to MAX_REDIRECT_HOPS.
/// Cross-domain redirects are surfaced instead of followed.
async fn follow_safe_redirects(
    client: &reqwest::Client,
    url: &str,
) -> std::result::Result<FollowResult, String> {
    let mut current_url = url.to_string();

    for _ in 0..MAX_REDIRECT_HOPS {
        let response = client
            .get(&current_url)
            .header("Accept", "text/markdown, text/html, */*")
            .header("User-Agent", "agentwerk/web_fetch")
            .send()
            .await
            .map_err(|e| format!("Fetch failed: {e}"))?;

        let status = response.status().as_u16();
        if !is_redirect(status) {
            return Ok(FollowResult::Ok(response));
        }

        let location = response
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok())
            .ok_or("Redirect missing Location header")?;

        let redirect_url = resolve_redirect_location(&current_url, location);

        if is_same_origin(&current_url, &redirect_url) {
            current_url = redirect_url;
        } else {
            return Ok(FollowResult::CrossDomain {
                original_url: url.to_string(),
                redirect_url,
                status,
            });
        }
    }

    Err(format!("Too many redirects (exceeded {MAX_REDIRECT_HOPS})"))
}

fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 307 | 308)
}

/// Allows redirects that keep the same scheme, port, and host (ignoring www. prefix).
/// Rejects cross-domain redirects and targets with embedded credentials.
fn is_same_origin(original_url: &str, redirect_url: &str) -> bool {
    let Some(orig) = parse_origin(original_url) else {
        return false;
    };
    let Some(redir) = parse_origin(redirect_url) else {
        return false;
    };

    orig.scheme == redir.scheme && orig.port == redir.port && orig.bare_host() == redir.bare_host()
}

struct UrlOrigin {
    scheme: String,
    host: String,
    port: String,
}

impl UrlOrigin {
    fn bare_host(&self) -> &str {
        self.host.strip_prefix("www.").unwrap_or(&self.host)
    }
}

fn parse_origin(url: &str) -> Option<UrlOrigin> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split('/').next().unwrap_or(rest);

    if authority.contains('@') {
        return None;
    }

    let (host, port) = authority
        .split_once(':')
        .map(|(h, p)| (h.to_string(), p.to_string()))
        .unwrap_or_else(|| (authority.to_string(), String::new()));

    Some(UrlOrigin {
        scheme: scheme.to_string(),
        host,
        port,
    })
}

/// Resolve a possibly-relative Location header against the request URL.
fn resolve_redirect_location(base_url: &str, location: &str) -> String {
    if location.starts_with("http://") || location.starts_with("https://") {
        return location.to_string();
    }

    let Some(scheme_end) = base_url.find("://") else {
        return location.to_string();
    };
    let origin_end = base_url[scheme_end + 3..]
        .find('/')
        .map_or(base_url.len(), |j| scheme_end + 3 + j);

    if location.starts_with('/') {
        format!("{}{location}", &base_url[..origin_end])
    } else {
        let dir_end = base_url.rfind('/').unwrap_or(origin_end);
        format!("{}/{location}", &base_url[..dir_end])
    }
}

// -- URL validation -----------------------------------------------------------

fn validate_url(url: &str) -> std::result::Result<String, String> {
    if url.len() > MAX_URL_LENGTH {
        return Err(format!(
            "URL too long: {} chars (max {MAX_URL_LENGTH})",
            url.len()
        ));
    }

    let (scheme, rest) = url.split_once("://").ok_or("Invalid URL: missing scheme")?;
    if !matches!(scheme, "http" | "https") {
        return Err(format!("Unsupported scheme: {scheme}"));
    }

    let authority = rest.split('/').next().unwrap_or(rest);
    if authority.contains('@') {
        return Err("URLs with embedded credentials are not allowed".into());
    }

    let host = authority.split(':').next().unwrap_or(authority);
    if host.is_empty() {
        return Err("URL must have a hostname".into());
    }
    if host.split('.').count() < 2 {
        return Err("URL must have a publicly resolvable hostname".into());
    }

    if scheme == "http" {
        return Ok(format!("https://{rest}"));
    }
    Ok(url.to_string())
}

// -- HTML-to-text -------------------------------------------------------------

fn strip_html(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_invisible_block = false;
    let mut chars = html.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '<' {
            let lookahead: String = chars.clone().take(10).collect();
            let lower = lookahead.to_lowercase();

            if lower.starts_with("script") || lower.starts_with("style") {
                in_invisible_block = true;
            } else if lower.starts_with("/script") || lower.starts_with("/style") {
                in_invisible_block = false;
            }
            in_tag = true;
            continue;
        }
        if ch == '>' {
            in_tag = false;
            continue;
        }
        if in_tag || in_invisible_block {
            continue;
        }
        if ch == '&' {
            text.push_str(&decode_html_entity(&mut chars));
            continue;
        }
        text.push(ch);
    }

    collapse_whitespace(&text)
}

fn decode_html_entity(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
    let mut name = String::new();
    for _ in 0..10 {
        match chars.peek() {
            Some(&';') => {
                chars.next();
                return resolve_named_entity(&name);
            }
            Some(&c) if c.is_alphanumeric() || c == '#' => {
                name.push(c);
                chars.next();
            }
            _ => break,
        }
    }
    format!("&{name}")
}

fn resolve_named_entity(name: &str) -> String {
    match name {
        "amp" => "&".into(),
        "lt" => "<".into(),
        "gt" => ">".into(),
        "quot" => "\"".into(),
        "apos" => "'".into(),
        "nbsp" => " ".into(),
        s if s.starts_with("#x") || s.starts_with("#X") => decode_numeric_entity(&s[2..], 16, name),
        s if s.starts_with('#') => decode_numeric_entity(&s[1..], 10, name),
        _ => format!("&{name};"),
    }
}

fn decode_numeric_entity(digits: &str, radix: u32, original: &str) -> String {
    u32::from_str_radix(digits, radix)
        .ok()
        .and_then(char::from_u32)
        .map(|c| c.to_string())
        .unwrap_or_else(|| format!("&{original};"))
}

fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut consecutive_blanks = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            consecutive_blanks += 1;
            if consecutive_blanks <= 1 {
                result.push('\n');
            }
        } else {
            consecutive_blanks = 0;
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(trimmed);
        }
    }

    result.trim().to_string()
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- URL validation -------------------------------------------------------

    #[test]
    fn validate_url_valid_https() {
        let result = validate_url("https://example.com/page");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "https://example.com/page");
    }

    #[test]
    fn validate_url_upgrades_http() {
        let result = validate_url("http://example.com/page");
        assert!(result.is_ok());
        assert!(result.unwrap().starts_with("https://"));
    }

    #[test]
    fn validate_url_accepts_port() {
        let result = validate_url("https://example.com:8080/page");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "https://example.com:8080/page");
    }

    #[test]
    fn validate_url_accepts_query_and_fragment() {
        assert!(validate_url("https://example.com/page?q=1&b=2#section").is_ok());
    }

    #[test]
    fn validate_url_rejects_no_host() {
        assert!(validate_url("https://").is_err());
    }

    #[test]
    fn validate_url_rejects_empty_host() {
        let err = validate_url("https:///path").unwrap_err();
        assert!(err.contains("hostname"));
    }

    #[test]
    fn validate_url_rejects_single_label_host() {
        let err = validate_url("https://localhost/page").unwrap_err();
        assert!(err.contains("publicly resolvable"));
    }

    #[test]
    fn validate_url_rejects_credentials() {
        let err = validate_url("https://user:pass@example.com").unwrap_err();
        assert!(err.contains("credentials"));
    }

    #[test]
    fn validate_url_rejects_too_long() {
        let long = format!("https://example.com/{}", "a".repeat(MAX_URL_LENGTH));
        let err = validate_url(&long).unwrap_err();
        assert!(err.contains("too long"));
    }

    #[test]
    fn validate_url_rejects_ftp() {
        let err = validate_url("ftp://example.com/file").unwrap_err();
        assert!(err.contains("Unsupported scheme"));
    }

    // -- Redirect safety ------------------------------------------------------

    #[test]
    fn redirect_same_host_permitted() {
        assert!(is_same_origin(
            "https://example.com/a",
            "https://example.com/b"
        ));
    }

    #[test]
    fn redirect_www_add_permitted() {
        assert!(is_same_origin(
            "https://example.com/a",
            "https://www.example.com/b"
        ));
    }

    #[test]
    fn redirect_www_remove_permitted() {
        assert!(is_same_origin(
            "https://www.example.com/a",
            "https://example.com/b"
        ));
    }

    #[test]
    fn redirect_path_change_permitted() {
        assert!(is_same_origin(
            "https://example.com/old/path",
            "https://example.com/new/path"
        ));
    }

    #[test]
    fn redirect_query_change_permitted() {
        assert!(is_same_origin(
            "https://example.com/page",
            "https://example.com/page?redirected=true"
        ));
    }

    #[test]
    fn redirect_cross_domain_rejected() {
        assert!(!is_same_origin(
            "https://example.com/a",
            "https://evil.com/b"
        ));
    }

    #[test]
    fn redirect_subdomain_rejected() {
        assert!(!is_same_origin(
            "https://sub.example.com/a",
            "https://example.com/b"
        ));
    }

    #[test]
    fn redirect_protocol_change_rejected() {
        assert!(!is_same_origin(
            "https://example.com/a",
            "http://example.com/b"
        ));
    }

    #[test]
    fn redirect_port_change_rejected() {
        assert!(!is_same_origin(
            "https://example.com:443/a",
            "https://example.com:8080/b"
        ));
    }

    #[test]
    fn redirect_with_credentials_rejected() {
        assert!(!is_same_origin(
            "https://example.com/a",
            "https://user:pass@example.com/b"
        ));
    }

    // -- Redirect resolution --------------------------------------------------

    #[test]
    fn resolve_absolute_redirect() {
        assert_eq!(
            resolve_redirect_location("https://example.com/a", "https://other.com/b"),
            "https://other.com/b"
        );
    }

    #[test]
    fn resolve_relative_redirect_absolute_path() {
        assert_eq!(
            resolve_redirect_location("https://example.com/old/page", "/new/page"),
            "https://example.com/new/page"
        );
    }

    #[test]
    fn resolve_relative_redirect_relative_path() {
        assert_eq!(
            resolve_redirect_location("https://example.com/old/page", "other"),
            "https://example.com/old/other"
        );
    }

    #[test]
    fn resolve_redirect_preserves_query() {
        assert_eq!(
            resolve_redirect_location("https://example.com/old", "/new?q=1&b=2"),
            "https://example.com/new?q=1&b=2"
        );
    }

    #[test]
    fn resolve_redirect_no_path_in_base() {
        assert_eq!(
            resolve_redirect_location("https://example.com", "/page"),
            "https://example.com/page"
        );
    }

    // -- HTML stripping -------------------------------------------------------

    #[test]
    fn strip_html_basic() {
        let text = strip_html("<html><body><h1>Hello</h1><p>World</p></body></html>");
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn strip_html_entities() {
        assert_eq!(
            strip_html("Tom &amp; Jerry &lt;3 &gt; &quot;cats&quot;"),
            "Tom & Jerry <3 > \"cats\""
        );
    }

    #[test]
    fn strip_html_numeric_entities() {
        assert_eq!(strip_html("&#65;&#x42;"), "AB");
    }

    #[test]
    fn strip_html_removes_script() {
        let text = strip_html("<p>Before</p><script>alert('xss')</script><p>After</p>");
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn strip_html_removes_style() {
        let text = strip_html("<style>body { color: red; }</style><p>Content</p>");
        assert!(text.contains("Content"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn strip_html_self_closing_tags() {
        let text = strip_html("before<br/><img src='x'/>after");
        assert!(text.contains("before"));
        assert!(text.contains("after"));
        assert!(!text.contains("img"));
    }

    #[test]
    fn strip_html_nested_tags() {
        assert_eq!(
            strip_html("<div><p><b>deep text</b></p></div>"),
            "deep text"
        );
    }

    #[test]
    fn strip_html_attributes_stripped() {
        assert_eq!(
            strip_html(r#"<a href="https://example.com" class="link">click here</a>"#),
            "click here"
        );
    }

    #[test]
    fn strip_html_preserves_plain_text() {
        let text = "no tags here, just plain text";
        assert_eq!(strip_html(text), text);
    }

    #[test]
    fn collapse_whitespace_limits_blank_lines() {
        assert_eq!(
            collapse_whitespace("line1\n\n\n\n\nline2"),
            "line1\n\nline2"
        );
    }
}
