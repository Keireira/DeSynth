use crate::catalog::{Catalog, default_steps_for_lightning, nonempty_file};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::path::Path;

pub use crate::images::RestorationParams as RestoreParams;

#[derive(Clone, Debug, Serialize)]
pub struct ProcessParams {
    #[serde(flatten)]
    pub restoration: RestoreParams,
    pub denoise: f64,
    pub steps: u32,
    pub passes: u32,
    pub seed: Option<u64>,
    pub transformer: Option<String>,
    pub preview: bool,
    pub keep_intermediate: bool,
    pub device: String,
    pub offload: String,
    pub precision: String,
    pub lightning: Option<String>,
    pub lightning_scale: f64,
}

pub fn parse_restore(raw: &Value) -> Result<RestoreParams, String> {
    let fields = object(raw)?;
    let restore_mode = text(fields, "restore_mode", "gaussian")?;
    if !["gaussian", "edge"].contains(&restore_mode.as_str()) {
        return Err("restore_mode must be gaussian or edge.".into());
    }
    Ok(RestoreParams {
        restore_sigma: number(fields, "restore_sigma", 1.95, 0.1, 10.0)?,
        unsharp: number(fields, "unsharp", 0.0, 0.0, 3.0)?,
        restore_mode,
        restore: boolean(fields, "restore", true)?,
    })
}

pub fn parse_process(raw: &Value, default_steps: u32) -> Result<ProcessParams, String> {
    let fields = object(raw)?;
    let device = text(fields, "device", "auto")?;
    let cuda_device = device.strip_prefix("cuda:").is_some_and(|index| {
        !index.is_empty() && index.bytes().all(|character| character.is_ascii_digit())
    });
    if !["auto", "cpu", "mps"].contains(&device.as_str()) && !cuda_device {
        return Err("Unknown device. Choose auto, cpu, mps or an available cuda:N device.".into());
    }
    let offload = choice(
        fields,
        "offload",
        "auto",
        &["auto", "sequential", "model", "none"],
    )?;
    let precision = choice(
        fields,
        "precision",
        "auto",
        &["auto", "bfloat16", "float16", "float32"],
    )?;
    let denoise = number(fields, "denoise", 0.25, 0.01, 1.0)?;
    let steps = integer(fields, "steps", default_steps, 1, 64)?;
    if effective_img2img_steps(steps, denoise) < 1 {
        return Err(
            "These settings produce no processing steps. Increase steps or denoise.".into(),
        );
    }
    Ok(ProcessParams {
        restoration: parse_restore(raw)?,
        denoise,
        steps,
        passes: integer(fields, "passes", 1, 1, 8)?,
        seed: seed(fields)?,
        transformer: optional_text(fields, "transformer")?,
        preview: boolean(fields, "preview", false)?,
        keep_intermediate: boolean(fields, "keep_intermediate", false)?,
        device,
        offload,
        precision,
        lightning: optional_text(fields, "lightning")?,
        lightning_scale: number(fields, "lightning_scale", 0.8, 0.01, 2.0)?,
    })
}

pub fn selected_default_steps(raw: &Value, catalog: &Catalog) -> u32 {
    let selected = raw.get("lightning").and_then(Value::as_str);
    if let Some(adapter) = catalog
        .lightning_adapters
        .iter()
        .find(|adapter| Some(adapter.path.as_str()) == selected)
    {
        return default_steps_for_lightning(&adapter.path);
    }
    catalog.default_steps()
}

pub fn select_models(
    params: &mut ProcessParams,
    catalog: &Catalog,
    root: &Path,
) -> Result<Vec<String>, String> {
    if let Some(requested) = &params.transformer
        && !catalog.models.iter().any(|model| model.path == *requested)
    {
        return Err("Select a GGUF model from the available models. Use Refresh in Settings to update the list.".into());
    }
    let adapter = params
        .lightning
        .as_deref()
        .or_else(|| catalog.default_lightning());
    let selected = catalog.lightning_adapters.iter().find(|item| Some(item.path.as_str()) == adapter)
        .ok_or("Select a Lightning adapter from the available adapters. Use Refresh in Settings to update the list.")?;
    if !nonempty_file(&root.join(&selected.path)) {
        return Err("The selected Lightning adapter was moved or removed. Use Refresh in Settings to update the list.".into());
    }
    params.lightning = Some(selected.path.clone());
    let candidates = match &params.transformer {
        Some(path) => vec![path.clone()],
        None => catalog.automatic_model_paths(root),
    };
    if candidates.is_empty()
        || candidates
            .iter()
            .any(|path| !nonempty_file(&root.join(path)))
    {
        return Err("The selected GGUF model was moved or removed. Use Refresh in Settings to update the list.".into());
    }
    Ok(candidates)
}

pub fn resolve_runtime(params: &ProcessParams, runtime: &Value) -> Result<Value, String> {
    if runtime.get("ready").and_then(Value::as_bool) != Some(true) {
        return Err("Devices are still being checked. Please retry in a moment.".into());
    }
    let device_id = if params.device == "auto" {
        runtime
            .get("default_device")
            .and_then(Value::as_str)
            .unwrap_or("cpu")
    } else {
        &params.device
    };
    let devices = runtime.get("devices").and_then(Value::as_array);
    let capability = devices
        .and_then(|devices| {
            devices
                .iter()
                .find(|device| device.get("id").and_then(Value::as_str) == Some(device_id))
        })
        .ok_or_else(|| {
            format!("Device {device_id} is unavailable. Select a device from the available list.")
        })?;
    let name = capability
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(device_id);
    let device_type = capability
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("cpu");
    let offload_modes = strings(capability, "offload_modes");
    let offload = if params.offload == "auto" {
        match device_type {
            "cuda" => "sequential",
            "mps" => "model",
            _ => "none",
        }
    } else {
        &params.offload
    };
    if !offload_modes.contains(&offload) {
        return Err(format!(
            "{offload} offload is not supported on {name}. Choose {}.",
            offload_modes.join(", ")
        ));
    }
    let precisions = strings(capability, "precisions");
    let precision = if params.precision == "auto" {
        if precisions.contains(&"bfloat16") {
            "bfloat16"
        } else {
            "float32"
        }
    } else {
        &params.precision
    };
    if !precisions.contains(&precision) {
        if precision == "float16" {
            return Err("Qwen-Image can overflow in float16 and produce a black image. Choose bfloat16 (if available) or float32.".into());
        }
        return Err(format!(
            "{precision} precision is not supported on {name}. Choose {}.",
            precisions.join(", ")
        ));
    }
    Ok(
        json!({"device": device_id, "offload": offload, "precision": precision, "device_name": name}),
    )
}

pub fn effective_img2img_steps(steps: u32, denoise: f64) -> u32 {
    let initial = (f64::from(steps) * denoise).min(f64::from(steps));
    steps - (f64::from(steps) - initial).max(0.0) as u32
}

fn object(raw: &Value) -> Result<&Map<String, Value>, String> {
    raw.as_object()
        .ok_or_else(|| "params must be an object.".into())
}

fn number(
    fields: &Map<String, Value>,
    name: &str,
    default: f64,
    low: f64,
    high: f64,
) -> Result<f64, String> {
    let value = match fields.get(name) {
        None => default,
        Some(value) => value
            .as_f64()
            .ok_or_else(|| format!("{name}: a finite number is required."))?,
    };
    if !value.is_finite() {
        return Err(format!("{name}: a finite number is required."));
    }
    if !(low..=high).contains(&value) {
        return Err(format!(
            "{name}: expected a value between {low} and {high}."
        ));
    }
    Ok(value)
}

fn integer(
    fields: &Map<String, Value>,
    name: &str,
    default: u32,
    low: u32,
    high: u32,
) -> Result<u32, String> {
    let value = number(
        fields,
        name,
        f64::from(default),
        f64::from(low),
        f64::from(high),
    )?;
    if value.fract() != 0.0 {
        return Err(format!(
            "{name}: expected a value between {low} and {high}."
        ));
    }
    Ok(value as u32)
}

fn boolean(fields: &Map<String, Value>, name: &str, default: bool) -> Result<bool, String> {
    match fields.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("{name} must be a boolean.")),
    }
}

fn text(fields: &Map<String, Value>, name: &str, default: &str) -> Result<String, String> {
    match fields.get(name) {
        None => Ok(default.into()),
        Some(value) => value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("{name} must be a string.")),
    }
}

fn optional_text(fields: &Map<String, Value>, name: &str) -> Result<Option<String>, String> {
    match fields.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(|text| Some(text.into()))
            .ok_or_else(|| format!("{name} must be a string or null.")),
    }
}

fn choice(
    fields: &Map<String, Value>,
    name: &str,
    default: &str,
    options: &[&str],
) -> Result<String, String> {
    let value = text(fields, name, default)?;
    if !options.contains(&value.as_str()) {
        return Err(format!("{name} must be {}.", options.join(", ")));
    }
    Ok(value)
}

fn seed(fields: &Map<String, Value>) -> Result<Option<u64>, String> {
    let Some(value) = fields.get("seed").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    const MAX: u64 = (1_u64 << 63) - 9;
    if let Some(seed) = value.as_u64().filter(|seed| *seed <= MAX) {
        return Ok(Some(seed));
    }
    if let Some(seed) = value.as_f64()
        && seed.is_finite()
        && seed >= 0.0
        && seed < (1_u64 << 63) as f64
        && seed.fract() == 0.0
    {
        let seed = seed as u64;
        if seed <= MAX {
            return Ok(Some(seed));
        }
    }
    Err(format!("seed: expected an integer between 0 and {MAX}."))
}

fn strings<'a>(value: &'a Value, name: &str) -> Vec<&'a str> {
    value
        .get(name)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_selected_lightning_profile() {
        let params = parse_process(&json!({}), 32).unwrap();
        assert_eq!(params.steps, 32);
        assert_eq!(effective_img2img_steps(params.steps, params.denoise), 8);
        assert_eq!(serde_json::to_value(params).unwrap()["restore_sigma"], 1.95);
    }

    #[test]
    fn rejects_wrong_types_and_out_of_bounds_inputs() {
        for value in [
            json!({"steps": true}),
            json!({"steps": 8.5}),
            json!({"seed": true}),
            json!({"seed": "42"}),
            json!({"restore": 1}),
            json!({"preview": "false"}),
            json!({"transformer": []}),
            json!({"restore_mode": "other"}),
            json!({"device": "cuda:"}),
            json!({"denoise": 0}),
            json!({"passes": 9}),
        ] {
            assert!(parse_process(&value, 32).is_err(), "accepted {value}");
        }
        assert!(parse_process(&Value::Null, 32).is_err());
        assert!(parse_process(&json!({"seed": (1_u64 << 63) - 8}), 32).is_err());
        assert_eq!(
            parse_process(&json!({"seed": (1_u64 << 63) - 9}), 32)
                .unwrap()
                .seed,
            Some((1_u64 << 63) - 9)
        );
    }

    #[test]
    fn runtime_defaults_use_device_capabilities_and_reject_fp16() {
        let runtime = json!({"ready":true,"default_device":"cuda:0","devices":[
            {"id":"cpu","name":"CPU","type":"cpu","precisions":["float32"],"offload_modes":["none"]},
            {"id":"cuda:0","name":"GPU","type":"cuda","precisions":["bfloat16","float32"],"offload_modes":["sequential","model","none"]}
        ]});
        let mut params = parse_process(&json!({}), 32).unwrap();
        let resolved = resolve_runtime(&params, &runtime).unwrap();
        assert_eq!(resolved["precision"], "bfloat16");
        assert_eq!(resolved["offload"], "sequential");
        params.precision = "float16".into();
        assert!(
            resolve_runtime(&params, &runtime)
                .unwrap_err()
                .contains("black image")
        );
        params.device = "cpu".into();
        params.precision = "auto".into();
        assert_eq!(
            resolve_runtime(&params, &runtime).unwrap()["offload"],
            "none"
        );
        params.offload = "model".into();
        assert!(resolve_runtime(&params, &runtime).is_err());
    }
}
