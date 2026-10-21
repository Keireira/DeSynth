use std::fmt;

#[derive(Clone, Debug)]
pub struct ServiceError {
    pub status: u16,
    pub message: String,
}

impl ServiceError {
    pub fn bad(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }
    pub fn missing(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
        }
    }
    pub fn busy(message: impl Into<String>) -> Self {
        Self {
            status: 409,
            message: message.into(),
        }
    }
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: 503,
            message: message.into(),
        }
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output.write_str(&self.message)
    }
}
impl std::error::Error for ServiceError {}
impl From<anyhow::Error> for ServiceError {
    fn from(error: anyhow::Error) -> Self {
        tracing::error!(%error, "Image operation failed");
        Self {
            status: 500,
            message: "The image operation failed. Check the DeSynth terminal.".to_owned(),
        }
    }
}
impl From<std::io::Error> for ServiceError {
    fn from(error: std::io::Error) -> Self {
        anyhow::Error::from(error).into()
    }
}

pub type ServiceResult<T> = Result<T, ServiceError>;
