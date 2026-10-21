"""Ключ и сертификат сервера для связи вне дома (ТЗ, ОД-3а: шифрование).

Сертификат самоподписанный: домен и удостоверяющий центр не нужны. Плеер при сопряжении дома
запоминает отпечаток (SHA-256 сертификата) и дальше верит только ему — подменить сервер
другим сертификатом нельзя. Ключ лежит в каталоге данных с правами «только владелец».
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import ssl
from dataclasses import dataclass
from pathlib import Path

VALID_YEARS = 10


@dataclass
class Identity:
    cert: Path
    key: Path
    fingerprint: str  # hex SHA-256 сертификата (DER), как его считает плеер


def _paths(data_dir: Path) -> tuple[Path, Path]:
    d = data_dir / "tls"
    return d / "server.crt", d / "server.key"


def ensure(data_dir: Path, name: str) -> Identity:
    """Создаёт ключ ECDSA P-256 и сертификат при первом вызове, потом только читает."""
    cert, key = _paths(data_dir)
    if not (cert.exists() and key.exists()):
        _create(cert, key, name)
    return Identity(cert, key, fingerprint_of(cert))


def fingerprint_of(cert: Path) -> str:
    der = ssl.PEM_cert_to_DER_cert(cert.read_text())
    return hashlib.sha256(der).hexdigest()


def _create(cert: Path, key: Path, name: str) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cert.parent, 0o700)
    except OSError:
        pass
    pk = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"Arx Nova {name}"[:64])])
    now = dt.datetime.now(dt.timezone.utc)
    crt = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(pk.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=365 * VALID_YEARS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(pk, hashes.SHA256())
    )
    tmp = key.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption()))
    os.replace(tmp, key)
    cert.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
