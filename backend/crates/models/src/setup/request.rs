use anyhow::{Context, Result};
use reqwest::{RequestBuilder, Response};
use std::time::Duration;

const MAX_ATTEMPTS: u64 = 3;

/// Retry idempotent model GETs before handing their response body to the caller.
pub(super) async fn send(request: RequestBuilder) -> Result<Response> {
    let mut attempt = 1;
    loop {
        let next = request
            .try_clone()
            .context("Model download requests must be repeatable GET requests")?;
        match next.send().await {
            Ok(response) => return Ok(response.error_for_status()?),
            Err(error) => {
                let retryable = error.is_connect() || error.is_timeout() || error.is_request();
                if !retryable || attempt == MAX_ATTEMPTS {
                    return Err(error).with_context(|| {
                        format!(
                            "Network request failed after {attempt} attempt(s); rerun make models to resume"
                        )
                    });
                }
                let delay = Duration::from_secs(attempt);
                eprintln!(
                    "Network request failed ({attempt}/{MAX_ATTEMPTS}): {error}. Retrying in {}s...",
                    delay.as_secs()
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests;
