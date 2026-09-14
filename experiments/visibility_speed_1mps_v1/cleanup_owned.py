"""Find detached simulator descendants by this run's exact environment identity."""
import os,signal,time
from pathlib import Path

def owned_pids(directory):
 target=str(Path(directory).resolve());out=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
   if env.get(b'CAPTURE_SWEEP_DIR',b'').decode()==target:out.append(int(p.name))
  except (OSError,ValueError):continue
 return out

def cleanup(directory,exclude=()):
 victims=[p for p in owned_pids(directory) if p not in exclude]
 for sig in [signal.SIGTERM,signal.SIGKILL]:
  for pid in victims:
   try:os.kill(pid,sig)
   except ProcessLookupError:pass
  if sig==signal.SIGTERM:time.sleep(1)
 return victims
