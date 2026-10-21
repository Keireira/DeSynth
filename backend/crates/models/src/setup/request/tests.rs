use super::send;
use reqwest::{Client, StatusCode};
use std::{io, net::SocketAddr, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::mpsc,
    task::JoinHandle,
    time::timeout,
};

#[derive(Clone, Copy)]
enum Reply {
    DropFirst,
    PlaintextTls,
    NotFound,
}

struct Server {
    address: SocketAddr,
    requests: mpsc::UnboundedReceiver<Vec<u8>>,
    task: Option<JoinHandle<()>>,
}

impl Server {
    async fn start(reply: Reply) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, requests) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let mut attempt = 0;
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let request = timeout(Duration::from_secs(2), async {
                    if matches!(reply, Reply::PlaintextTls) {
                        return read_tls_record(&mut stream).await;
                    }
                    read_headers(&mut stream).await
                })
                .await
                .expect("The client must start its request promptly")
                .unwrap();
                sender.send(request).unwrap();
                attempt += 1;
                if matches!(reply, Reply::DropFirst) && attempt == 1 {
                    continue;
                }

                let response = if matches!(reply, Reply::NotFound) {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .as_slice()
                } else {
                    b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nready"
                        .as_slice()
                };
                stream.write_all(response).await.unwrap();
            }
        });
        Self {
            address,
            requests,
            task: Some(task),
        }
    }

    fn url(&self, scheme: &str) -> String {
        format!("{scheme}://{}/model", self.address)
    }

    async fn stop(mut self) -> Vec<Vec<u8>> {
        let task = self.task.take().unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let mut requests = Vec::new();
        while let Ok(request) = self.requests.try_recv() {
            requests.push(request);
        }
        requests
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

async fn read_headers(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut headers = Vec::new();
    let mut buffer = [0; 1024];
    while headers.len() < 16 * 1024 {
        let length = stream.read(&mut buffer).await?;
        if length == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        headers.extend_from_slice(&buffer[..length]);
        if headers.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
            return Ok(headers);
        }
    }
    Err(io::Error::from(io::ErrorKind::InvalidData))
}

async fn read_tls_record(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut header = [0; 5];
    stream.read_exact(&mut header).await?;
    let length = u16::from_be_bytes([header[3], header[4]]);
    let mut record = vec![0; 5 + usize::from(length)];
    record[..5].copy_from_slice(&header);
    stream.read_exact(&mut record[5..]).await?;
    Ok(record)
}

fn client() -> Client {
    Client::builder()
        .no_proxy()
        .retry(reqwest::retry::never())
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap()
}

#[tokio::test]
async fn dropped_connection_retries_with_the_original_range_header() {
    let server = Server::start(Reply::DropFirst).await;
    let request = client().get(server.url("http")).header("range", "bytes=3-");
    let response = timeout(Duration::from_secs(12), send(request))
        .await
        .expect("Transport retries must be bounded")
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), "ready");

    let requests = server.stop().await;
    assert_eq!(requests.len(), 2);
    for request in requests {
        let headers = String::from_utf8(request).unwrap().to_ascii_lowercase();
        assert!(headers.lines().any(|line| line == "range: bytes=3-"));
    }
}

#[tokio::test]
async fn invalid_tls_content_type_stops_after_three_attempts_and_keeps_the_source() {
    let server = Server::start(Reply::PlaintextTls).await;
    let error = timeout(
        Duration::from_secs(12),
        send(client().get(server.url("https"))),
    )
    .await
    .expect("Failed TLS handshakes must exhaust a bounded retry budget")
    .unwrap_err();
    let requests = server.stop().await;
    assert_eq!(requests.len(), 3);
    assert!(
        requests
            .iter()
            .all(|request| request.first() == Some(&0x16))
    );
    assert!(error.to_string().contains("after 3 attempt(s)"));
    assert!(format!("{error:#}").contains("InvalidContentType"));
    let source = error.downcast_ref::<reqwest::Error>().unwrap();
    assert!(source.is_connect());
}

#[tokio::test]
async fn http_not_found_is_returned_without_retrying() {
    let server = Server::start(Reply::NotFound).await;
    let error = timeout(
        Duration::from_secs(12),
        send(client().get(server.url("http"))),
    )
    .await
    .expect("A terminal HTTP status must return promptly")
    .unwrap_err();
    assert_eq!(server.stop().await.len(), 1);
    let source = error.downcast_ref::<reqwest::Error>().unwrap();
    assert_eq!(source.status(), Some(StatusCode::NOT_FOUND));
}
