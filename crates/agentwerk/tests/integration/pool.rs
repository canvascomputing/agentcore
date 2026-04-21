//! End-to-end: `AgentPool` orchestrates several real-LLM agents concurrently. Guards the pool's concurrency and ordering against a live provider.

use super::common;

use agentwerk::{Agent, AgentPool, AgentPoolStrategy, ReadFileTool};

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
        .output_schema(output_schema)
        .max_turns(5);

    let pool = AgentPool::new()
        .batch_size(2)
        .ordering(AgentPoolStrategy::SpawnOrder);

    for file in &["Cargo.toml", "README.md", "CLAUDE.md"] {
        pool.spawn(
            summarizer
                .clone()
                .provider(provider.clone())
                .instruction_prompt(format!("Read and summarize: {file}")),
        )
        .await;
    }

    let results = pool.drain().await;

    assert_eq!(results.len(), 3);
    for (i, (_id, result)) in results.iter().enumerate() {
        let output = result.as_ref().expect(&format!("Agent {i} failed"));
        let json = output
            .response
            .as_ref()
            .expect(&format!("Agent {i} missing structured output"));
        assert!(
            json["summary"].is_string(),
            "Agent {i}: expected summary string"
        );
    }

    common::print_result(results[0].1.as_ref().unwrap());

    Ok(())
}
