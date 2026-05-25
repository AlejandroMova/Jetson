"""
Debug: test Digest auth reusing the same TCP connection.
"""
import socket, re, hashlib

host, port = '192.168.10.68', 554
user, password = 'admin', 'S1=me77SL'
url = f'rtsp://{host}:{port}/cam/realmonitor?channel=1&subtype=0'

def recv(s):
    """Lee respuesta RTSP del socket hasta encontrar \\r\\n\\r\\n o timeout."""
    data = b''
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            data += chunk
            if b'\r\n\r\n' in data: break
    except (socket.timeout, OSError):
        pass
    return data.decode(errors='ignore')

def make_req(auth_header='', cseq=1):
    """Construye una petición RTSP DESCRIBE en bytes con cabecera de auth opcional."""
    req = (f'DESCRIBE {url} RTSP/1.0\r\n'
           f'CSeq: {cseq}\r\nUser-Agent: test/1.0\r\nAccept: application/sdp\r\n')
    if auth_header:
        req += f'Authorization: {auth_header}\r\n'
    return (req + '\r\n').encode()

def build_digest(user, password, method, uri, realm, nonce):
    """Construye la cabecera Authorization: Digest RFC 2617 para las credenciales y nonce dados."""
    ha1 = hashlib.md5(f'{user}:{realm}:{password}'.encode()).hexdigest()
    ha2 = hashlib.md5(f'{method}:{uri}'.encode()).hexdigest()
    resp = hashlib.md5(f'{ha1}:{nonce}:{ha2}'.encode()).hexdigest()
    return (f'Digest username="{user}", realm="{realm}", nonce="{nonce}", '
            f'uri="{uri}", algorithm="MD5", response="{resp}"')

print('=== SAME connection (like GStreamer) ===')
with socket.create_connection((host, port), timeout=5) as s:
    s.settimeout(5)

    # Step 1: unauthenticated
    s.sendall(make_req(cseq=1))
    text = recv(s)
    print('Step 1:', repr(text))

    m = re.search(r'WWW-Authenticate:\s*(.+)', text)
    www_auth = m.group(1).strip()
    realm = re.search(r'realm="([^"]+)"', www_auth).group(1).strip()
    nonce = re.search(r'nonce="([^"]+)"', www_auth).group(1).strip()
    print(f'realm={repr(realm)}  nonce={repr(nonce)}')

    # Step 2: authenticated — SAME connection
    auth = build_digest(user, password, 'DESCRIBE', url, realm, nonce)
    print(f'Auth: {auth}')
    s.sendall(make_req(auth, cseq=2))
    text2 = recv(s)
    print('Step 2:', repr(text2[:300]))
