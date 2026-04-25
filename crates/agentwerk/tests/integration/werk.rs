//! End-to-end: `Werk::produce` runs several real-LLM agents concurrently. Guards line capping and result correlation against a live provider.

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
        .model_name(&model)
        .tool(ReadFileTool)
        .schema(output_schema)
        .max_turns(5);

    let files = ["Cargo.toml", "README.md", "CLAUDE.md"];
    let agents = files.iter().map(|file| {
        summarizer
            .clone()
            .name(format!("summarize-{file}"))
            .provider(provider.clone())
            .task(format!("Read and summarize: {file}"))
    });

    let results = Werk::new().lines(2).hire_and_fire(agents).await;

    assert_eq!(results.len(), files.len());
    for result in &results {
        let output = result.as_ref().expect("agent failed");
        let json = output.response.as_ref().expect("missing structured output");
        assert!(
            json["summary"].is_string(),
            "{}: expected summary string",
            output.name
        );
    }

    common::print_result(results[0].as_ref().unwrap());

    Ok(())
}
