//! HTTP fetch boundary for `atenia download`.
//!
//! The orchestration in [`super::run_download`] depends on this
//! trait, not on `ureq` directly, so tests can substitute a fake
//! fetcher that serves bytes from memory. The real implementation
//! is [`UreqFetcher`].
//!
//! Contract is deliberately minimal:
//!   - one method, `fetch_to_writer`,
//!   - synchronous,
//!   - errors are string-typed and bubble up as `E-DOWNLOAD-NETWORK`
//!     via the CLI error layer — the fetch implementation does not
//!     classify network errors any further.

use std::io::Write;
use std::time::Duration;

/// One-method HTTP fetch boundary.
///
/// Implementations stream the response body into `sink`. The CLI
/// layer is responsible for opening the sink (typically a
/// `<file>.partial` file), atomic rename on success, and cleanup
/// on failure.
pub trait HttpFetcher {
    /// Fetch `url` into `sink`. Returns the number of bytes
    /// written. Network, TLS, DNS and timeout faults all collapse
    /// to the `Err(String)` arm; the orchestrator turns that into
    /// `E-DOWNLOAD-NETWORK`.
    fn fetch_to_writer(&self, url: &str, sink: &mut dyn Write) -> Result<u64, String>;
}

/// Production fetcher backed by `ureq` + rustls. One simple retry
/// with exponential backoff is wired in here, not at the
/// orchestrator level, so the trait stays single-method.
pub struct UreqFetcher {
    agent: ureq::Agent,
    /// Number of retries (so total attempts = retries + 1).
    retries: u32,
    /// Sleep before each retry; doubles on each subsequent retry.
    backoff_base: Duration,
}

impl Default for UreqFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl UreqFetcher {
    /// Production defaults: 30 s connect+read timeout per attempt,
    /// 1 retry with a 2 s backoff, follow up to 10 redirects (HF
    /// LFS serves the largest weight files via a 302 to a CDN URL).
    pub fn new() -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(30))
            .timeout_read(Duration::from_secs(60))
            .timeout_write(Duration::from_secs(60))
            .redirects(10)
            .user_agent(concat!(
                "atenia/",
                env!("CARGO_PKG_VERSION"),
                " (+https://github.com/AteniaEngine/ateniaengine)"
            ))
            .build();
        Self {
            agent,
            retries: 1,
            backoff_base: Duration::from_secs(2),
        }
    }
}

impl HttpFetcher for UreqFetcher {
    fn fetch_to_writer(&self, url: &str, sink: &mut dyn Write) -> Result<u64, String> {
        let mut last_err: Option<String> = None;
        for attempt in 0..=self.retries {
            if attempt > 0 {
                let pause = self.backoff_base * (1 << (attempt - 1));
                std::thread::sleep(pause);
            }
            match self.agent.get(url).call() {
                Ok(resp) => {
                    let mut reader = resp.into_reader();
                    match std::io::copy(&mut reader, sink) {
                        Ok(n) => return Ok(n),
                        Err(e) => last_err = Some(format!("read failed: {e}")),
                    }
                }
                Err(ureq::Error::Status(code, resp)) => {
                    // 4xx will not improve with a retry; bail
                    // immediately so the user gets a precise error
                    // instead of waiting through the backoff.
                    let detail = format!(
                        "HTTP {code} for {url}: {}",
                        resp.into_string().unwrap_or_default()
                    );
                    if (400..500).contains(&code) {
                        return Err(detail);
                    }
                    last_err = Some(detail);
                }
                Err(ureq::Error::Transport(t)) => {
                    last_err = Some(format!("transport error: {t}"));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| "unknown network error".into()))
    }
}

pub mod test_support {
    //! In-memory fake fetcher for integration tests. Lives here
    //! (next to the trait) instead of inside `tests/` so the
    //! integration tests in `tests/cli_download_test.rs` can use
    //! it through the public crate surface. Dead code in release
    //! builds — the orchestrator only ever instantiates
    //! [`super::UreqFetcher`].

    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub struct FakeFetcher {
        /// URL → bytes to write (or `Err` to simulate a network
        /// failure). The first matching entry wins.
        responses: HashMap<String, Result<Vec<u8>, String>>,
        /// URLs that were actually requested, in call order.
        calls: Mutex<Vec<String>>,
    }

    impl FakeFetcher {
        pub fn new() -> Self {
            Self {
                responses: HashMap::new(),
                calls: Mutex::new(Vec::new()),
            }
        }

        pub fn with_body(mut self, url: impl Into<String>, body: impl Into<Vec<u8>>) -> Self {
            self.responses.insert(url.into(), Ok(body.into()));
            self
        }

        pub fn with_failure(mut self, url: impl Into<String>, err: impl Into<String>) -> Self {
            self.responses.insert(url.into(), Err(err.into()));
            self
        }

        pub fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl HttpFetcher for FakeFetcher {
        fn fetch_to_writer(&self, url: &str, sink: &mut dyn Write) -> Result<u64, String> {
            self.calls.lock().unwrap().push(url.to_string());
            match self.responses.get(url) {
                Some(Ok(body)) => {
                    sink.write_all(body)
                        .map_err(|e| format!("sink write failed: {e}"))?;
                    Ok(body.len() as u64)
                }
                Some(Err(msg)) => Err(msg.clone()),
                None => Err(format!("FakeFetcher: no canned response for {url}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::FakeFetcher;
    use super::*;

    #[test]
    fn fake_fetcher_serves_canned_body() {
        let f = FakeFetcher::new().with_body("https://x/a", b"hello".to_vec());
        let mut buf = Vec::new();
        let n = f.fetch_to_writer("https://x/a", &mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(buf, b"hello");
        assert_eq!(f.calls(), vec!["https://x/a".to_string()]);
    }

    #[test]
    fn fake_fetcher_propagates_failure() {
        let f = FakeFetcher::new().with_failure("https://x/a", "boom");
        let mut buf = Vec::new();
        let err = f.fetch_to_writer("https://x/a", &mut buf).unwrap_err();
        assert!(err.contains("boom"));
    }

    #[test]
    fn fake_fetcher_unknown_url_errors() {
        let f = FakeFetcher::new();
        let mut buf = Vec::new();
        let err = f.fetch_to_writer("https://x/missing", &mut buf).unwrap_err();
        assert!(err.contains("no canned response"));
    }
}
