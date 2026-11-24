"""Start an isolated smoke worker in a dedicated cgroup; keep its status and UMA guard together.
Conservative system-wide admission signal, NOT an exact service peak or kernel-enforced UMA limit.
"""
import argparse,json,os,re,signal,subprocess,sys,time
from pathlib import Path

def available():return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024

def run(args):
    if not args.command:raise ValueError('missing command')
    procgroup=Path('/proc/self/cgroup').read_text(); suffix=next(x[3:] for x in procgroup.splitlines() if x.startswith('0::'))
    if not re.fullmatch(r'.*/aibook-slot-smoke-[a-zA-Z0-9-]+\.service',suffix):raise ValueError('must run in a dedicated aibook-slot-smoke unit')
    cap=int(args.cap_gb*1_000_000_000); margin=int(args.margin_gb*1_000_000_000); idle=min(available() for _ in range(8)); peak=0;floor=idle-cap+margin
    if idle<cap+margin:raise RuntimeError('insufficient starting headroom')
    cg=Path('/sys/fs/cgroup')/suffix.lstrip('/');
    def sample():
        a=available();c=int((cg/'memory.current').read_text());swap=int((cg/'memory.swap.current').read_text());return a,c,swap
    a,c,s=sample();print(json.dumps(dict(event='baseline',available=a,cgroup=c,stop_floor=floor)),flush=True)
    p=subprocess.Popen(args.command);deadline=time.monotonic()+args.seconds;st=0
    while p.poll() is None:
        a,c,s=sample();peak=max(peak,idle-a);now=time.monotonic()
        if now>=st:print(json.dumps(dict(event='sample',available=a,peak_available_drop=peak,cgroup=c,swap=s)),flush=True);st=now+2
        if a<floor or c>cap-margin or s or now>=deadline:
            print(json.dumps(dict(event='guard-stop',available=a,cgroup=c,swap=s,timeout=now>=deadline)),flush=True);p.send_signal(signal.SIGTERM);return 124 # systemd ends remaining unit members
        time.sleep(.1)
    print(json.dumps(dict(event='exit',returncode=p.returncode,peak_available_drop=peak)),flush=True);return p.returncode
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,default=900);ap.add_argument('--cap-gb',type=float,default=24.0);ap.add_argument('--margin-gb',type=float,default=2.0);ap.add_argument('command',nargs=argparse.REMAINDER);sys.exit(run(ap.parse_args()))
