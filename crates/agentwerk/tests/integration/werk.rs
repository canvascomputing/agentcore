//! End-to-end: `Werk::work` runs several real-LLM agents concurrently. Guards line capping and result correlation against a live provider.

use super::common;

use agentwerk::tools::ReadFileTool;
use agentwerk::{Agent, Werk};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let output_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One-sentence summary"
            }
        },
        "required": ["summary"]
    });

    let summarizer = Agent::new()
        .model(&model)
        .tool(ReadFileTool)
        .contract(output_schema)
        .max_steps(5);

    let files = ["Cargo.toml", "README.md", "CLAUDE.md"];
    let agents = files.iter().map(|file| {
        summarizer
            .clone()
            .name(format!("summarize-{file}"))
            .provider(provider.clone())
            .work(format!("Read and summarize: {file}"))
    });

    let results = Werk::new().lines(2).work(agents).await;

    assert_eq!(results.len(), files.len());
    for file in files {
        let key = format!("summarize-{file}");
        let output = results
            .get(&key)
            .unwrap_or_else(|| panic!("missing result for {key}"))
            .as_ref()
            .expect("agent failed");
        let json = output.response.as_ref().expect("missing structured output");
        assert!(
            json["summary"].is_string(),
            "{}: expected summary string",
            output.name
        );
    }

    let any_output = results
        .values()
        .next()
        .expect("at least one result")
        .as_ref()
        .unwrap();
    common::print_result(any_output);

    Ok(())
}
