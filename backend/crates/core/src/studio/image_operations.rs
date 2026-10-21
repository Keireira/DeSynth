use super::{Studio, prune};
use crate::{
    assets::{Asset, Assets, StoredAsset},
    error::{ServiceError, ServiceResult},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use desynth_images as images;
use serde_json::Value;

impl Studio {
    pub async fn image(&self, id: &str) -> ServiceResult<(StoredAsset, image::RgbaImage)> {
        let asset = self.asset(id)?;
        let reader = asset.clone();
        let image = self.cpu_task(move || Ok(reader.read_image()?)).await?;
        Ok((asset, image))
    }

    pub(crate) fn asset(&self, id: &str) -> ServiceResult<StoredAsset> {
        self.store
            .lock()
            .assets
            .get(id, None)
            .map_err(|error| ServiceError::missing(error.to_string()))
    }

    pub async fn image_bytes(&self, id: &str, raw: bool) -> ServiceResult<Vec<u8>> {
        let asset = self.asset(id)?;
        self.cpu_task(move || {
            let image = asset.read_image()?;
            if raw {
                return Ok(image.into_raw());
            }
            Ok(images::encode_png(&image, asset.metadata.as_ref())?)
        })
        .await
    }

    pub async fn upload(&self, payload: Value) -> ServiceResult<Asset> {
        let role = match payload.get("role") {
            None => "source",
            Some(value) => value
                .as_str()
                .ok_or_else(|| ServiceError::bad("role must be source or result."))?,
        };
        if !matches!(role, "source" | "result") {
            return Err(ServiceError::bad("role must be source or result."));
        }
        let data = payload
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| ServiceError::bad("Image data is missing."))?;
        let encoded = if data.starts_with("data:") {
            data.split_once(";base64,")
                .ok_or_else(|| ServiceError::bad("A base64-encoded image is required."))?
                .1
        } else {
            data
        };
        if encoded.len() > 45 * 1024 * 1024 {
            return Err(ServiceError {
                status: 413,
                message: "Maximum file size: 32 MB.".to_owned(),
            });
        }
        let name = match payload.get("name") {
            None => "image.png",
            Some(value) => value
                .as_str()
                .ok_or_else(|| ServiceError::bad("name must be a string."))?,
        };
        let normalized = name.replace('\\', "/");
        let name = normalized.rsplit('/').next().unwrap_or("image.png");
        let name: String = name.chars().take(160).collect();
        let name = if name.is_empty() {
            "image.png".to_owned()
        } else {
            name
        };
        let encoded = encoded.to_owned();
        let role = role.to_owned();
        let directory = self.root.join("out/gui");
        drop(payload);
        let prepared = self
            .cpu_task(move || {
                let bytes = STANDARD
                    .decode(encoded)
                    .map_err(|_| ServiceError::bad("Invalid base64 image data."))?;
                if bytes.len() > 32 * 1024 * 1024 {
                    return Err(ServiceError {
                        status: 413,
                        message: "Maximum file size: 32 MB.".to_owned(),
                    });
                }
                let image = images::open(&bytes)
                    .map_err(|error| ServiceError::bad(format!("Cannot read image: {error}")))?;
                Ok(Assets::prepare(
                    &directory, &image, &role, &name, true, None,
                )?)
            })
            .await?;
        let mut store = self.store.lock();
        if *self.shutdown.borrow() {
            return Err(ServiceError::unavailable("The server is stopping."));
        }
        let asset = store.assets.insert(prepared.commit());
        if asset.kind == "source" {
            store.assets.latest_source = Some(asset.id.clone());
        }
        prune(&mut store, [asset.id.clone()]);
        Ok(asset)
    }

    pub async fn compare(&self, source: &str, result: &str) -> ServiceResult<Value> {
        let source = self.asset(source)?;
        let result = self.asset(result)?;
        self.cpu_task(move || {
            let source = source.read_image()?;
            let result = result.read_image()?;
            images::metrics(&source, &result).map_err(|error| ServiceError::bad(error.to_string()))
        })
        .await
    }
}
