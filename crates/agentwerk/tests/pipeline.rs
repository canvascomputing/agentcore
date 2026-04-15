mod common;

use agentwerk::{AgentBuilder, Pipeline, ReadFileTool};

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

    let mut pipeline = Pipeline::new().batch_size(2);

    for file in &["Cargo.toml", "README.md", "CLAUDE.md"] {
        pipeline.push(
            AgentBuilder::new()
                .provider(provider.clone())
                .model(&model)
                .instruction_prompt(format!("Read and summarize: {file}"))
                .tool(ReadFileTool)
                .output_schema(output_schema.clone())
                .max_turns(5)
        );
    }

    let results = pipeline.run().await;

    assert_eq!(results.len(), 3);
    for (i, result) in results.iter().enumerate() {
        let output = result.as_ref().expect(&format!("Agent {i} failed"));
        let json = output.response.as_ref().expect(&format!("Agent {i} missing structured output"));
        assert!(json["summary"].is_string(), "Agent {i}: expected summary string");
    }

    common::print_result(results[0].as_ref().unwrap());

    Ok(())
}
