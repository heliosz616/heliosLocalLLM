"""Project-local supervisor with explicit process ownership; no global service changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_profiles import load_catalog, atomic_json, ensure_catalog
sys.path.insert(0, str(ROOT.parent/'runtime'))
from gpu_lease import GpuLease

def runtime_command(cfg, manifest, slots, internal_key):
    command = [str(ROOT/'build'/cfg.get('runtime_directory','runtime')/'llama-server.exe'),
        '--model',str(ROOT/'models'/manifest['filename']),
        '--host','127.0.0.1','--port','8081','--alias',cfg['model'],
        '--api-key-file',str(internal_key),'--cors-origins','http://127.0.0.1:8081','--no-cors-credentials',
        '--jinja','--log-verbosity','4','--parallel','1','--ctx-size',str(cfg['context']),
        '--n-gpu-layers','99','--split-mode','layer','--device',cfg['devices'],
        '--tensor-split',cfg['tensor_split'],'--fit','off','--flash-attn','on',
        '--cache-type-k',cfg['cache_type_k'],'--cache-type-v',cfg['cache_type_v'],
        '--batch-size',str(cfg['batch']),'--ubatch-size',str(cfg['ubatch']),
        '--ctx-checkpoints',str(cfg['checkpoints']),'--checkpoint-min-step','1024',
        '--cache-ram','0','--no-context-shift','--slots','--metrics',
        '--slot-save-path',str(slots)+os.sep,'--temp','1.0','--top-p','0.95','--top-k','20']
    if cfg.get('spec_type'):
        if cfg['spec_type'] != 'draft-mtp':
            raise ValueError('Unsupported speculative decoding mode')
        command += ['--spec-type','draft-mtp','--spec-draft-n-max',str(cfg.get('spec_draft_n_max',1))]
        if 'spec_draft_p_min' in cfg:
            command += ['--spec-draft-p-min',str(cfg['spec_draft_p_min'])]
        command += ['--spec-draft-type-k',cfg.get('spec_cache_type_k',cfg['cache_type_k']),
                    '--spec-draft-type-v',cfg.get('spec_cache_type_v',cfg['cache_type_v'])]
    return command

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['run','stop','runtime'])
    parser.add_argument('--context',type=int)
    parser.add_argument('--devices')
    parser.add_argument('--split')
    parser.add_argument('--batch',type=int)
    parser.add_argument('--ubatch',type=int)
    parser.add_argument('--default-model')
    args = parser.parse_args()
    state = ROOT/'state'
    state.mkdir(exist_ok=True)
    stop = state/'stop.request'
    if args.action == 'stop':
        stop.touch()
        print('Requested supervisor shutdown')
        # Make `stop.ps1; start.ps1` safe: listener/process exit can precede
        # release of the ownership lock during supervisor teardown.
        import msvcrt
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            with (state/'supervisor.lock').open('a+b') as probe:
                probe.seek(0)
                try:
                    msvcrt.locking(probe.fileno(),msvcrt.LK_NBLCK,1)
                except OSError:
                    time.sleep(.1)
                    continue
                probe.seek(0)
                msvcrt.locking(probe.fileno(),msvcrt.LK_UNLCK,1)
                print('Supervisor stopped')
                return
        raise SystemExit('Shutdown timed out; inspect owned project processes before restarting')
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts'/'secure_state.ps1')],
                   check=True,stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    # OS lock is released by the OS even after a crash.
    import msvcrt
    owner = (state/'supervisor.lock').open('a+b')
    owner.seek(0)
    try:
        msvcrt.locking(owner.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        raise SystemExit('A project supervisor is already running')
    # Shared by the speech supervisor, before any GPU model is loaded.
    try:
        gpu_lease = GpuLease().acquire()
    except RuntimeError as exc:
        owner.close()
        raise SystemExit(str(exc))
    stop.unlink(missing_ok=True)
    cfgpath = ROOT/'config.local.json'
    if not cfgpath.exists():
        cfgpath.write_text((ROOT/'config.example.json').read_text())
    cfg = json.loads(cfgpath.read_text())
    catalog_file = ensure_catalog(ROOT, cfg)
    if args.default_model:
        catalog=load_catalog(ROOT,cfg,'models.local.json')
        if args.default_model not in catalog:
            raise SystemExit('Unknown default model')
        cfg=catalog[args.default_model]['config'].copy()
    if args.context:
        cfg['context'] = args.context
    if args.devices:
        cfg['devices'] = args.devices
    if args.split:
        cfg['tensor_split'] = args.split
    if args.batch:
        cfg['batch']=args.batch
    if args.ubatch:
        cfg['ubatch']=args.ubatch
    cfgpath.write_text(json.dumps(cfg,indent=2))
    keyfile = state/'api-key.txt'
    if not keyfile.exists():
        keyfile.write_text(secrets.token_urlsafe(36))
    internal_key=state/'internal-key.txt'
    if not internal_key.exists():
        internal_key.write_text(secrets.token_urlsafe(48))
    profiles = load_catalog(ROOT,cfg,catalog_file)
    slots = state/profiles[cfg['model']]['slots']
    slots.mkdir(exist_ok=True)
    manifest = json.loads((ROOT/'models'/profiles[cfg['model']]['manifest']).read_text())
    if manifest.get('verified_sha256') != manifest['sha256']:
        raise SystemExit('Run download_model.py to finish download and verification first')
    build=json.loads((ROOT/'reports'/'runtime-built.json').read_text(encoding='utf-8-sig'))
    for entry in build['files']:
        with (ROOT/'build'/'runtime'/entry['name']).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=entry['sha256']:
                raise SystemExit('Runtime changed after build; rebuild patched server before starting')
    from device_guard import parse_devices,resolve_devices
    inventory=subprocess.run([str(ROOT/'build/runtime/llama-server.exe'),'--list-devices'],
        cwd=ROOT/'build/runtime',capture_output=True,text=True,encoding='utf-8',errors='replace',
        timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    inventory.check_returncode()
    available=parse_devices(inventory.stdout+'\n'+inventory.stderr)
    try:
        cfg['devices'],cfg['device_identities']=resolve_devices(cfg['devices'],available,
            cfg.get('device_identities'),explicit=bool(args.devices))
    except ValueError as exc:
        raise SystemExit(str(exc))
    def align_vulkan_host(model_cfg):
        if 'GGML_VK_HOST_DEVICE' in model_cfg.get('runtime_env',{}):
            selected_vk=[item for item in model_cfg['devices'].split(',') if item.startswith('Vulkan')]
            if len(selected_vk)!=1:
                raise ValueError('Pinned Vulkan host tuning requires exactly one selected Vulkan device')
            model_cfg['runtime_env']=dict(model_cfg['runtime_env'],GGML_VK_HOST_DEVICE=selected_vk[0][6:])
    align_vulkan_host(cfg)
    cfgpath.write_text(json.dumps(cfg,indent=2))
    profiles[cfg['model']]['config']=cfg.copy()
    if catalog_file:
        atomic_json(ROOT/catalog_file,dict(models=profiles))
    (ROOT/'reports/device-selection.json').write_text(json.dumps(dict(
        available=available,selected=cfg['devices'],identities=cfg['device_identities']),indent=2))
    command = runtime_command(cfg, manifest, slots, internal_key)
    (state/'runtime-command.json').write_text(json.dumps(command,indent=2))
    log = (ROOT/'reports'/'runtime.log').open('ab',buffering=0)
    gateway_log = (ROOT/'reports'/'gateway.log').open('ab',buffering=0)
    children = []
    runtime_env=os.environ.copy()
    if cfg.get('profile_transfers'):
        runtime_env['LLM_SERVICE_PROFILE_TRANSFERS']='1'
    for name,value in cfg.get('runtime_env',{}).items():
        if not name.startswith(('GGML_CUDA_','GGML_VK_')) and name not in ('LLM_SERVICE_FINE_CHECKPOINTS','LLM_SERVICE_MTP_KV_ONLY','LLM_SERVICE_MTP_UBATCH','LLM_SERVICE_MTP_NO_PIPELINE','CUDA_MODULE_LOADING'):
            raise ValueError('Unsupported runtime tuning environment variable')
        runtime_env[name]=str(value)
    def halt(*_):
        stop.touch()
    signal.signal(signal.SIGINT,halt)
    signal.signal(signal.SIGTERM,halt)
    def launch(name):
        profile = profiles[name]
        model_cfg = profile['config'].copy()
        model_cfg['devices'], model_cfg['device_identities'] = resolve_devices(
            model_cfg['devices'],available,model_cfg.get('device_identities'))
        align_vulkan_host(model_cfg)
        model_manifest = json.loads((ROOT/'models'/profile['manifest']).read_text())
        model_path = ROOT/'models'/model_manifest['filename']
        if (model_manifest.get('verified_sha256') != model_manifest['sha256'] or
                not model_path.is_file() or model_path.stat().st_size != model_manifest['size']):
            raise RuntimeError('Model download is not verified')
        model_slots = state/profile['slots']
        model_slots.mkdir(exist_ok=True)
        runtime_dir=ROOT/'build'/model_cfg.get('runtime_directory','runtime')
        runtime_manifest=ROOT/'reports'/model_cfg.get('runtime_manifest','runtime-built.json')
        for entry in json.loads(runtime_manifest.read_text(encoding='utf-8-sig'))['files']:
            with (runtime_dir/entry['name']).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=entry['sha256']:
                    raise RuntimeError('Model runtime integrity check failed')
        cmd = runtime_command(model_cfg,model_manifest,model_slots,internal_key)
        env = os.environ.copy()
        if model_cfg.get('profile_transfers'):
            env['LLM_SERVICE_PROFILE_TRANSFERS']='1'
        for key,value in model_cfg.get('runtime_env',{}).items():
            if not key.startswith(('GGML_CUDA_','GGML_VK_')) and key not in ('LLM_SERVICE_FINE_CHECKPOINTS','LLM_SERVICE_MTP_KV_ONLY','LLM_SERVICE_MTP_UBATCH','LLM_SERVICE_MTP_NO_PIPELINE','CUDA_MODULE_LOADING'):
                raise ValueError('Unsupported runtime tuning environment variable')
            env[key]=str(value)
        atomic_json(state/'runtime-command.json',cmd)
        process = subprocess.Popen(cmd,cwd=runtime_dir,stdout=log,stderr=log,
            env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(process)
        return process
    def terminate(process):
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
    def ready(process):
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if stop.exists():
                raise RuntimeError('Shutdown requested')
            if process.poll() is not None:
                raise RuntimeError('Runtime exited during model load')
            try:
                probe=urllib.request.Request('http://127.0.0.1:8081/health',headers={'Authorization':'Bearer '+internal_key.read_text().strip()})
                with opener.open(probe,timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError('Model load timed out')
    def load_ready(name):
        for attempt in range(2):
            process=launch(name)
            try:
                ready(process)
                return process
            except RuntimeError:
                exited=process.poll() is not None
                terminate(process)
                children.remove(process)
                if stop.exists() or not exited or attempt:
                    raise
                print(f'Runtime exited during loading {name}; retrying once after full teardown',flush=True)
    try:
        (state/'model-switch-request.json').unlink(missing_ok=True)
        (state/'model-switch-result.json').unlink(missing_ok=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        runtime = load_ready(cfg['model'])
        active_model = cfg['model']
        atomic_json(state/'runtime-status.json',dict(model=active_model,status='ready',pid=runtime.pid))
        if args.action != 'runtime':
            gateway_env=os.environ.copy()
            gateway_env['LLM_SERVICE_CONFIG']=str(cfgpath)
            gateway_env.pop('LLM_SERVICE_MODEL_CATALOG',None)
            if catalog_file:
                gateway_env['LLM_SERVICE_MODEL_CATALOG']=catalog_file
            gateway = subprocess.Popen([sys.executable,'-m','uvicorn','gateway:app','--host',cfg['host'],
                '--port',str(cfg['port']),'--workers','1','--no-proxy-headers','--no-access-log'],
                cwd=ROOT,stdout=gateway_log,stderr=gateway_log,env=gateway_env,creationflags=subprocess.CREATE_NO_WINDOW)
            children.append(gateway)
        (state/'processes.json').write_text(json.dumps({'supervisor':os.getpid(),'children':[p.pid for p in children]}))
        print('Runtime ready; gateway starting' if args.action != 'runtime' else 'Runtime ready',flush=True)
        while not stop.exists():
            if args.action != 'runtime' and gateway.poll() is not None:
                raise RuntimeError('Gateway exited')
            request_path=state/'model-switch-request.json'
            if request_path.exists():
                request=json.loads(request_path.read_text())
                request_path.unlink()
                name=request['model']
                started=time.perf_counter()
                result=dict(id=request['id'],model=name,ok=False)
                print(f'Model switch requested: {name}',flush=True)
                try:
                    if name not in profiles:
                        raise ValueError('Unknown model')
                    # Only the gateway holding the generation lock writes requests.
                    atomic_json(state/'runtime-status.json',dict(model=name,status='loading'))
                    terminate(runtime)
                    if runtime in children:
                        children.remove(runtime)
                    active_model=None
                    runtime=load_ready(name)
                    active_model=name
                    result['ok']=True
                    atomic_json(state/'runtime-status.json',dict(model=name,status='ready',pid=runtime.pid))
                except Exception as exc:
                    terminate(runtime)
                    active_model=None
                    result['error']=type(exc).__name__
                    atomic_json(state/'runtime-status.json',dict(model=None,status='unavailable'))
                result['seconds']=time.perf_counter()-started
                print(f'Model switch finished: {name}, ok={result["ok"]}, seconds={result["seconds"]:.3f}',flush=True)
                atomic_json(state/'model-switch-result.json',result)
                atomic_json(state/'processes.json',{'supervisor':os.getpid(),'children':[p.pid for p in children if p.poll() is None]})
            if runtime.poll() is not None and active_model is not None:
                if args.action == 'runtime':
                    raise RuntimeError('Runtime exited')
                active_model=None
                atomic_json(state/'runtime-status.json',dict(model=None,status='unavailable'))
            time.sleep(0.1)
    finally:
        for p in reversed(children):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
        (state/'processes.json').unlink(missing_ok=True)
        owner.close()
        gpu_lease.close()

if __name__ == '__main__':
    main()
