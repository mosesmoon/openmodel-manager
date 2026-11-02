//! Network interfaces the local servers can be opened on, and the TCP
//! forwarders that do it. Servers always bind 127.0.0.1; each chosen address
//! gets a forwarder on the same port, so other computers reach exactly the
//! interfaces the user ticked.

use serde::Serialize;
use std::net::{IpAddr, SocketAddr};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NetAddr {
    /// Interface name ("Ethernet", "Wi-Fi", "enp5s0", ...).
    pub name: String,
    pub ip: String,
}

/// IPv4 addresses on this computer that other machines can reach: no
/// loopback, and no 169.254.x.x (an adapter that got no address).
pub fn lan_addresses() -> Vec<NetAddr> {
    let nets = sysinfo::Networks::new_with_refreshed_list();
    let mut out: Vec<NetAddr> = nets
        .list()
        .iter()
        .flat_map(|(name, data)| {
            data.ip_networks().iter().filter_map(move |n| match n.addr {
                IpAddr::V4(v4) if usable(&v4) => Some(NetAddr { name: name.clone(), ip: v4.to_string() }),
                _ => None,
            })
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name).then(a.ip.cmp(&b.ip)));
    out.dedup();
    out
}

fn usable(ip: &std::net::Ipv4Addr) -> bool {
    !ip.is_loopback() && !ip.is_link_local() && !ip.is_unspecified()
}

/// Forward `ip:port` to `127.0.0.1:port`. Fails when the address is gone or
/// the port is taken there.
pub async fn forward(ip: IpAddr, port: u16) -> std::io::Result<JoinHandle<()>> {
    let listener = TcpListener::bind(SocketAddr::new(ip, port)).await?;
    Ok(tokio::spawn(async move {
        while let Ok((mut inbound, _)) = listener.accept().await {
            tokio::spawn(async move {
                if let Ok(mut outbound) = TcpStream::connect(("127.0.0.1", port)).await {
                    let _ = tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await;
                }
            });
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn skips_unreachable_addresses() {
        assert!(!usable(&"127.0.0.1".parse().unwrap()));
        assert!(!usable(&"169.254.3.4".parse().unwrap()));
        assert!(usable(&"192.168.0.10".parse().unwrap()));
        for a in lan_addresses() {
            assert!(usable(&a.ip.parse().unwrap()));
        }
    }

    #[tokio::test]
    async fn forwards_to_loopback() {
        // An echo server on 127.0.0.1, reached through 127.0.0.2 (Linux) or,
        // where that address does not exist, through the first LAN address.
        let echo = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = echo.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut s, _) = echo.accept().await.unwrap();
            let mut b = [0u8; 4];
            s.read_exact(&mut b).await.unwrap();
            s.write_all(&b).await.unwrap();
        });
        let candidates: Vec<IpAddr> = std::iter::once("127.0.0.2".parse().unwrap()).chain(lan_addresses().iter().map(|a| a.ip.parse().unwrap())).collect();
        let mut via = None;
        for ip in candidates {
            if let Ok(h) = forward(ip, port).await {
                via = Some((ip, h));
                break;
            }
        }
        let Some((ip, h)) = via else { return }; // no second address on this machine
        let mut c = TcpStream::connect((ip, port)).await.unwrap();
        c.write_all(b"ping").await.unwrap();
        let mut b = [0u8; 4];
        c.read_exact(&mut b).await.unwrap();
        assert_eq!(&b, b"ping");
        h.abort();
    }
}
