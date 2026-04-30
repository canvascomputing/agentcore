//! Multi-agent loop driver. Each agent runs in its own tokio task
//! against the shared `TicketSystem` reachable via `agent.runtime.system`.

use std::sync::Arc;
use std::time::Duration;

use crate::providers::types::StreamEvent;
use crate::providers::{ContentBlock, Message, ModelRequest};

use super::agent::Agent;
use super::policy::PolicyConform;
use super::tickets::Status;

const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Drive many agents against one shared `TicketSystem`. Each agent runs
/// in its own tokio task; locks on the system are held only around the
/// queue/metric ops and never across `provider.respond().await`, so
/// model calls really do overlap.
pub async fn run_main_loop(agents: Vec<Agent>) {
    let mut handles = Vec::with_capacity(agents.len());
    for agent in agents {
        handles.push(tokio::spawn(handle_tickets(agent)));
    }
    for h in handles {
        let _ = h.await;
    }
}

/// Per-agent drain. Picks the next Todo, processes it, repeats. When the
/// queue empties, idles on `IDLE_POLL_INTERVAL` until something arrives,
/// the cancel signal fires, or a policy hits. Callers stop the agent by
/// setting the `interrupt_signal` on the `TicketSystem`.
pub(super) async fn handle_tickets(agent: Agent) {
    loop {
        let work = {
            let mut sys = agent.runtime.system.lock().unwrap();
            if sys.is_interrupted() || sys.policy_violated() {
                return;
            }
            let max_request_tokens = sys.policies().max_request_tokens;
            sys.list_by_status(Status::Todo)
                .first()
                .map(|t| t.key.clone())
                .map(|key| {
                    sys.record_step(&agent.name);
                    let _ = sys.update_status(&key, Status::InProgress);
                    let prompt = sys
                        .get(&key)
                        .map(|t| format!("{}\n\n{}", t.summary, t.description));
                    (key, prompt, max_request_tokens)
                })
        };

        let Some((key, Some(prompt), max_request_tokens)) = work else {
            tokio::time::sleep(IDLE_POLL_INTERVAL).await;
            continue;
        };

        let request = ModelRequest {
            model: agent.runtime.model.clone(),
            system_prompt: agent.system_prompt(),
            messages: vec![Message::user(prompt)],
            tools: Vec::new(),
            max_request_tokens,
            tool_choice: None,
        };
        let result = agent
            .runtime
            .provider
            .respond(request, no_op_event_handler())
            .await;

        match result {
            Ok(response) => {
                let mut sys = agent.runtime.system.lock().unwrap();
                sys.record_request(&agent.name, &response.usage);
                let text = extract_text(&response.content);
                let _ = sys.add_comment(&key, agent.name.clone(), text);
                let _ = sys.update_status(&key, Status::Done);
            }
            Err(e) => {
                let mut sys = agent.runtime.system.lock().unwrap();
                sys.record_error(&agent.name, &key, &e);
                let _ = sys.update_status(&key, Status::Failed);
            }
        }
    }
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn no_op_event_handler() -> Arc<dyn Fn(StreamEvent) + Send + Sync> {
    Arc::new(|_| {})
}

#[cfg(test)]
mod tests {
    use super::super::agent::{Runtime, DEFAULT_BEHAVIOR};
    use super::super::tickets::{TicketSystem, TicketType};
    use super::*;
    use crate::providers::types::{ModelResponse, ResponseStatus};
    use crate::providers::{Provider, ProviderResult, TokenUsage};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    /// MockProvider that yields once before responding so concurrent
    /// agents interleave. Captures the most recent observed
    /// `system_prompt` and `max_request_tokens` for assertions.
    struct MockProvider {
        text: String,
        usage: TokenUsage,
        captured_prompt: Arc<Mutex<Option<String>>>,
        captured_max_tokens: Arc<Mutex<Option<u32>>>,
    }

    struct MockHandles {
        prompt: Arc<Mutex<Option<String>>>,
        max_tokens: Arc<Mutex<Option<u32>>>,
    }

    impl MockProvider {
        fn new(text: impl Into<String>) -> (Arc<Self>, MockHandles) {
            Self::with_usage(text, TokenUsage::default())
        }

        fn with_usage(text: impl Into<String>, usage: TokenUsage) -> (Arc<Self>, MockHandles) {
            let prompt = Arc::new(Mutex::new(None));
            let max_tokens = Arc::new(Mutex::new(None));
            let provider = Arc::new(Self {
                text: text.into(),
                usage,
                captured_prompt: Arc::clone(&prompt),
                captured_max_tokens: Arc::clone(&max_tokens),
            });
            (
                provider,
                MockHandles {
                    prompt,
                    max_tokens,
                },
            )
        }
    }

    impl Provider for MockProvider {
        fn respond(
            &self,
            request: ModelRequest,
            _on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
            *self.captured_prompt.lock().unwrap() = Some(request.system_prompt);
            *self.captured_max_tokens.lock().unwrap() = request.max_request_tokens;
            let text = self.text.clone();
            let usage = self.usage.clone();
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(ModelResponse {
                    content: vec![ContentBlock::Text { text }],
                    status: ResponseStatus::EndTurn,
                    usage,
                    model: "mock".into(),
                })
            })
        }
    }

    fn build(
        provider: Arc<dyn Provider>,
        system: Arc<Mutex<TicketSystem>>,
    ) -> Arc<Runtime> {
        Arc::new(Runtime {
            provider,
            model: "mock".into(),
            system,
        })
    }

    fn shared_system() -> (Arc<Mutex<TicketSystem>>, Arc<AtomicBool>) {
        let signal = Arc::new(AtomicBool::new(false));
        let system = TicketSystem::new().interrupt_signal(Arc::clone(&signal));
        (Arc::new(Mutex::new(system)), signal)
    }

    fn seed(system: &Arc<Mutex<TicketSystem>>, summary: &str) {
        system.lock().unwrap().create(
            summary.into(),
            String::new(),
            TicketType::Task,
            "tester".into(),
        );
    }

    /// Drive `agents` to completion. A watcher polls the queue and
    /// triggers the interrupt once nothing is `Todo` or `InProgress`,
    /// so the idle agents wake up and exit.
    async fn drain_then_interrupt(
        system: Arc<Mutex<TicketSystem>>,
        signal: Arc<AtomicBool>,
        agents: Vec<Agent>,
    ) {
        let watcher_signal = Arc::clone(&signal);
        let watcher_system = Arc::clone(&system);
        let watcher = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(20)).await;
                let sys = watcher_system.lock().unwrap();
                let pending = sys.list_by_status(Status::Todo).len()
                    + sys.list_by_status(Status::InProgress).len();
                if pending == 0 || sys.policy_violated() {
                    watcher_signal.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });
        run_main_loop(agents).await;
        let _ = watcher.await;
    }

    #[tokio::test]
    async fn single_agent_drains_queue() {
        let (provider, _) = MockProvider::new("ok");
        let (system, signal) = shared_system();
        seed(&system, "first");
        seed(&system, "second");

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime).role("you are a worker");
        drain_then_interrupt(Arc::clone(&system), signal, vec![agent]).await;

        let sys = system.lock().unwrap();
        assert_eq!(sys.steps(), 2);
        assert_eq!(sys.requests(), 2);
        let done = sys.list_by_status(Status::Done);
        assert_eq!(done.len(), 2);
        for t in done {
            let comment = t
                .comments
                .iter()
                .find(|c| c.author == "worker")
                .expect("worker comment");
            assert_eq!(comment.body, "ok");
        }
    }

    #[tokio::test]
    async fn interrupt_signal_stops_an_idle_agent() {
        let (provider, _) = MockProvider::new("ok");
        let (system, signal) = shared_system();
        signal.store(true, Ordering::Relaxed);

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime);
        run_main_loop(vec![agent]).await;

        let sys = system.lock().unwrap();
        assert_eq!(sys.steps(), 0);
        assert_eq!(sys.requests(), 0);
    }

    #[tokio::test]
    async fn max_steps_stops_processing_after_threshold() {
        let (provider, _) = MockProvider::new("ok");
        let signal = Arc::new(AtomicBool::new(false));
        let system = Arc::new(Mutex::new(
            TicketSystem::new()
                .interrupt_signal(Arc::clone(&signal))
                .max_steps(2),
        ));
        seed(&system, "a");
        seed(&system, "b");
        seed(&system, "c");

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime);
        drain_then_interrupt(Arc::clone(&system), signal, vec![agent]).await;

        let sys = system.lock().unwrap();
        assert_eq!(sys.list_by_status(Status::Done).len(), 2);
        assert_eq!(sys.list_by_status(Status::Todo).len(), 1);
    }

    #[tokio::test]
    async fn max_input_tokens_stops_processing_after_threshold() {
        let (provider, _) = MockProvider::with_usage(
            "ok",
            TokenUsage {
                input_tokens: 60,
                output_tokens: 0,
                ..TokenUsage::default()
            },
        );
        let signal = Arc::new(AtomicBool::new(false));
        let system = Arc::new(Mutex::new(
            TicketSystem::new()
                .interrupt_signal(Arc::clone(&signal))
                .max_input_tokens(100),
        ));
        seed(&system, "a");
        seed(&system, "b");
        seed(&system, "c");

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime);
        drain_then_interrupt(Arc::clone(&system), signal, vec![agent]).await;

        let sys = system.lock().unwrap();
        assert_eq!(sys.list_by_status(Status::Done).len(), 2);
        assert_eq!(sys.list_by_status(Status::Todo).len(), 1);
    }

    #[tokio::test]
    async fn system_prompt_includes_role_behavior_and_context() {
        let (provider, handles) = MockProvider::new("ok");
        let (system, signal) = shared_system();
        seed(&system, "task");

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime)
            .role("ROLE_TEXT")
            .behavior("BEHAVIOR_TEXT")
            .context("CONTEXT_TEXT");
        drain_then_interrupt(Arc::clone(&system), signal, vec![agent]).await;

        let prompt = handles
            .prompt
            .lock()
            .unwrap()
            .clone()
            .expect("prompt captured");
        let role_pos = prompt.find("ROLE_TEXT").expect("role present");
        let behavior_pos = prompt.find("BEHAVIOR_TEXT").expect("behavior present");
        let context_pos = prompt.find("CONTEXT_TEXT").expect("context present");
        assert!(role_pos < behavior_pos);
        assert!(behavior_pos < context_pos);
    }

    #[tokio::test]
    async fn system_prompt_falls_back_to_default_behavior_when_unset() {
        let (provider, handles) = MockProvider::new("ok");
        let (system, signal) = shared_system();
        seed(&system, "task");

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime).role("ROLE_TEXT");
        drain_then_interrupt(Arc::clone(&system), signal, vec![agent]).await;

        let prompt = handles
            .prompt
            .lock()
            .unwrap()
            .clone()
            .expect("prompt captured");
        assert!(prompt.contains("ROLE_TEXT"));
        assert!(prompt.contains(DEFAULT_BEHAVIOR));
    }

    #[tokio::test]
    async fn max_request_tokens_is_forwarded_to_the_provider_request() {
        let (provider, handles) = MockProvider::new("ok");
        let signal = Arc::new(AtomicBool::new(false));
        let system = Arc::new(Mutex::new(
            TicketSystem::new()
                .interrupt_signal(Arc::clone(&signal))
                .max_request_tokens(256),
        ));
        seed(&system, "task");

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime);
        drain_then_interrupt(Arc::clone(&system), signal, vec![agent]).await;

        assert_eq!(*handles.max_tokens.lock().unwrap(), Some(256));
    }

    #[tokio::test]
    async fn run_main_loop_drives_multiple_agents_in_parallel() {
        let (provider, _) = MockProvider::new("done");
        let (system, signal) = shared_system();
        seed(&system, "a");
        seed(&system, "b");
        seed(&system, "c");
        seed(&system, "d");

        let runtime = build(provider, Arc::clone(&system));
        let alice = Agent::new("alice", Arc::clone(&runtime));
        let bob = Agent::new("bob", Arc::clone(&runtime));
        drain_then_interrupt(Arc::clone(&system), signal, vec![alice, bob]).await;

        let sys = system.lock().unwrap();
        assert_eq!(sys.list_by_status(Status::Done).len(), 4);
        assert_eq!(sys.steps(), 4);
        assert_eq!(sys.requests(), 4);
        let authors: Vec<&str> = sys
            .list_by_status(Status::Done)
            .iter()
            .flat_map(|t| t.comments.iter().map(|c| c.author.as_str()))
            .collect();
        assert!(authors.contains(&"alice"));
        assert!(authors.contains(&"bob"));
    }

    #[tokio::test]
    async fn idle_agent_picks_up_a_late_arriving_ticket() {
        let (provider, _) = MockProvider::new("done");
        let (system, signal) = shared_system();

        let runtime = build(provider, Arc::clone(&system));
        let agent = Agent::new("worker", runtime);
        let task = tokio::spawn(run_main_loop(vec![agent]));

        tokio::time::sleep(Duration::from_millis(50)).await;
        seed(&system, "late");

        // Watcher: interrupt once the late ticket has been processed.
        let watcher_signal = Arc::clone(&signal);
        let watcher_system = Arc::clone(&system);
        let watcher = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(20)).await;
                let sys = watcher_system.lock().unwrap();
                if sys.list_by_status(Status::Done).len() == 1 {
                    watcher_signal.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });

        let _ = task.await;
        let _ = watcher.await;

        let sys = system.lock().unwrap();
        assert_eq!(sys.list_by_status(Status::Done).len(), 1);
        assert_eq!(sys.requests(), 1);
    }
}
