use std::{sync::Arc, time::Duration};

use anyhow::{Result, bail};
use desynth_core::studio::Studio;
use serde_json::Value;

pub(super) async fn wait_for_probe(studio: &Arc<Studio>) -> Result<()> {
    for _ in 0..240 {
        let state = studio.state();
        if state["runtime"]["ready"] == true && state["runtime"]["refreshing"] != true {
            if state["ready"] != true {
                let missing = state["missing"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                bail!("Inference is unavailable ({missing}). Run make setup first.");
            }
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    bail!("Timed out while checking the inference worker")
}

pub(super) async fn wait_for_job(studio: &Studio, id: &str) -> Result<Value> {
    let mut previous_stage = String::new();
    loop {
        let job = studio.job(id)?;
        match job["status"].as_str() {
            Some("done") => return Ok(job),
            Some("error") => bail!("{}", job["error"].as_str().unwrap_or("Processing failed")),
            _ => {}
        }
        if let Some(stage) = job["stage"]
            .as_str()
            .filter(|stage| *stage != previous_stage)
        {
            println!("  {stage}");
            previous_stage = stage.to_owned();
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}
