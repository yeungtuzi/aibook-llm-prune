"""Build a JSONL external n-gram corpus (one JSON string per document)."""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

MAN_COMMANDS = """systemctl journalctl apt apt-get dpkg snap ip ss netstat nmcli resolvectl
mount umount lsblk blkid df du findmnt chmod chown tar rsync ssh scp ssh-keygen sshd
ufw iptables nft fail2ban rsyslog logrotate cron crontab systemd-run systemd-analyze
timedatectl hostnamectl localectl loginctl update-grub grub-install dmesg lsof strace
tcpdump curl wget git docker podman nginx apache2 mysql psql python3 pip sed awk grep
xargs tee sort uniq cut tr bash sh env export alias source sudo su useradd usermod
groupadd passwd visudo fdisk parted mkfs fsck smartctl sensors top htop ps kill pkill
nice ionice ulimit sysctl modprobe lsmod dkms update-initramfs apt-mark add-apt-repository
do-release-upgrade""".split()

def read_text(path, limit=20000):
    try:
        data = path.read_text(errors='replace')
    except Exception:
        return None
    return data[:limit] if data else None

def man_doc(cmd):
    try:
        p = subprocess.run(['man', '-P', 'cat', cmd], capture_output=True, text=True,
                           timeout=12, env={**os.environ, 'MANWIDTH': '110'})
    except Exception:
        return None
    out = p.stdout
    if not out or len(out) < 200:
        return None
    m = re.search(r'^\s*EXAMPLES?\s*$', out, re.M)
    if m:
        out = out[m.start():m.start() + 6000]
    else:
        out = out[:6000]
    return out

def docs(src, model):
    tldr = Path(src)/'tldr-main'/'pages'
    if tldr.is_dir():
        for p in sorted(tldr.rglob('*.md')):
            t = read_text(p, 12000)
            if t:
                yield f'# {p.relative_to(tldr)}\n{t}'
    cheat = Path(src)/'cheatsheets-master'
    if cheat.is_dir():
        for p in sorted(cheat.rglob('*')):
            if p.is_file() and p.suffix in ('', '.md', '.txt', '.rst'):
                t = read_text(p, 12000)
                if t:
                    yield t
    for p in sorted(Path('/usr/lib/systemd/system').glob('*.service')):
        t = read_text(p, 12000)
        if t:
            yield f'# {p.name}\n{t}'
    for p in [Path('/etc/fstab'), Path('/etc/ssh/sshd_config'), Path('/etc/apt/sources.list'),
              Path('/etc/systemd/resolved.conf'), Path('/etc/systemd/journald.conf')]:
        t = read_text(p, 12000)
        if t:
            yield f'# {p}\n{t}'
    for cmd in MAN_COMMANDS:
        t = man_doc(cmd)
        if t:
            yield f'# man {cmd}\n{t}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=Path.home()/'aibook-bench/ngram-src')
    ap.add_argument('--out', type=Path, default=Path.home()/'aibook-bench/ngram-corpus.jsonl')
    ap.add_argument('--model', default='/home/lf/models/gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead')
    ap.add_argument('--max-tokens', type=int, default=7_000_000)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    total = docs_n = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as f:
        for d in docs(args.src, args.model):
            if not d.strip():
                continue
            n = len(tok.encode(d, add_special_tokens=False))
            if n == 0:
                continue
            if total + n > args.max_tokens:
                break
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            total += n
            docs_n += 1
    print(json.dumps({'out': str(args.out), 'docs': docs_n, 'tokens': total,
                      'bytes': args.out.stat().st_size, 'max_tokens': args.max_tokens}))

if __name__ == '__main__':
    main()
