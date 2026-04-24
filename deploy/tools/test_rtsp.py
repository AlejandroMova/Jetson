"""
Debug RTSP Digest auth against Dahua DVR.
Tests two uri variants: full URL vs path-only.
"""
import socket, re, hashlib

host, port = '192.168.10.68', 554
user, password = 'admin', 'S1=me77SL'
full_url  = f'rtsp://{host}:{port}/cam/realmonitor?channel=1&subtype=0'
path_only = '/cam/realmonitor?channel=1&subtype=0'

def recv(s):
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

def make_req(url, auth_header='', cseq=1):
    req = (f'DESCRIBE {url} RTSP/1.0\r\n'
           f'CSeq: {cseq}\r\nUser-Agent: test/1.0\r\nAccept: application/sdp\r\n')
    if auth_header:
        req += f'Authorization: {auth_header}\r\n'
    return (req + '\r\n').encode()

def build_digest(user, password, method, uri, realm, nonce):
    ha1 = hashlib.md5(f'{user}:{realm}:{password}'.encode()).hexdigest()
    ha2 = hashlib.md5(f'{method}:{uri}'.encode()).hexdigest()
    resp = hashlib.md5(f'{ha1}:{nonce}:{ha2}'.encode()).hexdigest()
    return (f'Digest username="{user}", realm="{realm}", nonce="{nonce}", '
            f'uri="{uri}", algorithm="MD5", response="{resp}"')

# ── Step 1: get challenge ─────────────────────────────────────────────────────
print('=== Step 1: unauthenticated DESCRIBE ===')
s = socket.create_connection((host, port), timeout=5)
s.settimeout(5)
s.sendall(make_req(full_url, cseq=1))
text = recv(s)
s.close()
print(repr(text))

m = re.search(r'WWW-Authenticate:\s*(.+)', text)
if not m:
    print('ERROR: no WWW-Authenticate in response')
    exit(1)

www_auth = m.group(1).strip()
realm = re.search(r'realm="([^"]+)"', www_auth).group(1).strip()
nonce = re.search(r'nonce="([^"]+)"', www_auth).group(1).strip()
print(f'\nrealm = {repr(realm)}')
print(f'nonce = {repr(nonce)}')
print(f'pass  = {repr(password)}')

# ── Step 2a: full URL as digest-uri ──────────────────────────────────────────
print('\n=== Step 2a: auth with FULL URL as uri ===')
auth_a = build_digest(user, password, 'DESCRIBE', full_url, realm, nonce)
print(f'Header: {auth_a}')
s2a = socket.create_connection((host, port), timeout=5)
s2a.settimeout(5)
s2a.sendall(make_req(full_url, auth_a, cseq=2))
text2a = recv(s2a)
s2a.close()
print(repr(text2a))

# ── Step 2b: path-only as digest-uri (some Dahua firmware) ───────────────────
# Need fresh nonce — get a new challenge first
print('\n=== Getting fresh nonce for Step 2b ===')
s_fresh = socket.create_connection((host, port), timeout=5)
s_fresh.settimeout(5)
s_fresh.sendall(make_req(full_url, cseq=1))
text_fresh = recv(s_fresh)
s_fresh.close()
m2 = re.search(r'nonce="([^"]+)"', text_fresh)
nonce2 = m2.group(1).strip() if m2 else nonce

print(f'nonce2 = {repr(nonce2)}')
print('\n=== Step 2b: auth with PATH-ONLY as uri ===')
auth_b = build_digest(user, password, 'DESCRIBE', path_only, realm, nonce2)
print(f'Header: {auth_b}')
s2b = socket.create_connection((host, port), timeout=5)
s2b.settimeout(5)
s2b.sendall(make_req(full_url, auth_b, cseq=2))
text2b = recv(s2b)
s2b.close()
print(repr(text2b))
