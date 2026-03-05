# Inferctl Server Management Commands

## Overview

`inferctl` now includes comprehensive server management commands for starting, stopping, monitoring, and debugging InferFlux servers.

---

## Commands Reference

### `inferctl server start`

Start the InferFlux server in background with PID tracking.

```bash
inferctl server start [--config PATH] [--no-wait]
```

**Options:**
- `--config PATH` - Server config file (default: `~/.inferflux/config.yaml`)
- `--no-wait` - Don't wait for server to be ready (return immediately after fork)

**Features:**
- ✅ Starts server in background (fork + exec)
- ✅ PID tracking in `~/.inferflux/server.pid`
- ✅ Logs written to `~/.inferflux/logs/server.log`
- ✅ Health check waits for server to be ready (default: 30s timeout)
- ✅ Color-coded success messages

**Example:**
```bash
# Start with default config
inferctl server start

# Start with specific config
inferctl server start --config config/server.cuda.yaml

# Start with native CUDA backend
INFERFLUX_NATIVE_CUDA_EXECUTOR=native inferctl server start --config config/server.cuda.yaml

# Start without waiting for ready check
inferctl server start --no-wait
```

**Output:**
```
Starting InferFlux server (PID: 45093)...
Config: config/server.cuda.yaml
Logs: /home/user/.inferflux/logs/server.log
Waiting for server to be ready.....
✓ Server is ready!
  API: http://127.0.0.1:8080
  Health: http://127.0.0.1:8080/healthz
  Metrics: http://127.0.0.1:8080/metrics
```

---

### `inferctl server stop`

Stop the running InferFlux server gracefully.

```bash
inferctl server stop [--force]
```

**Options:**
- `--force` - Use SIGKILL instead of SIGTERM (2s timeout vs 10s)

**Features:**
- ✅ Graceful shutdown with SIGTERM
- ✅ Force kill with `--force` option
- ✅ Automatic PID file cleanup
- ✅ Waits for process to terminate

**Example:**
```bash
# Graceful stop
inferctl server stop

# Force kill (for stuck servers)
inferctl server stop --force
```

**Output:**
```
Stopping server (PID: 45093)....
✓ Server stopped
```

---

### `inferctl server status`

Show server status, PID, and health information.

```bash
inferctl server status [--verbose]
```

**Options:**
- `--verbose`, `-v` - Show detailed health and model information

**States:**
- 🟢 **RUNNING** (green) - Server is running and responsive
- 🔴 **STOPPED** (red) - Server is not running
- 🟡 **CRASHED** (red) - PID file exists but process not running (auto-cleanup)

**Example:**
```bash
# Basic status
inferctl server status

# Verbose status with health check
inferctl server status --verbose
```

**Output (basic):**
```
Server: RUNNING
PID: 45093
Config: /home/user/.inferflux/config.yaml
Logs: /home/user/.inferflux/logs/server.log
```

**Output (verbose):**
```
Server: RUNNING
PID: 45093
Config: /home/user/.inferflux/config.yaml
Logs: /home/user/.inferflux/logs/server.log

Checking server health...
Health: OK
Model Ready: yes

Loaded Models:
  - tinyllama [cuda] ✓
```

---

### `inferctl server restart`

Restart the server (stop + start).

```bash
inferctl server restart [--config PATH] [--no-wait]
```

**Options:**
- Same as `server start`

**Example:**
```bash
# Restart with same config
inferctl server restart

# Restart with new config
inferctl server restart --config config/server.cuda.yaml

# Quick restart (don't wait for ready)
inferctl server restart --no-wait
```

---

### `inferctl server logs`

Show server logs with optional tail/follow.

```bash
inferctl server logs [--tail N]
```

**Options:**
- `--tail N` - Show last N lines without following (default: follow mode)

**Example:**
```bash
# Follow logs (default, like tail -f)
inferctl server logs

# Show last 50 lines
inferctl server logs --tail 50

# Show last 500 lines
inferctl server logs --tail 500
```

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| **PID File** | `~/.inferflux/server.pid` | Tracks running server process |
| **Log File** | `~/.inferflux/logs/server.log` | Server stdout/stderr output |
| **Config** | `~/.inferflux/config.yaml` | Default server configuration |

---

## Environment Variables

Server management commands respect the following environment variables:

```bash
# Server configuration
export INFERFLUX_MODEL_PATH=models/model.gguf
export INFERCTL_API_KEY=dev-key-123

# Backend selection
export INFERFLUX_NATIVE_CUDA_EXECUTOR=native  # native, delegate, direct_llama
export INFERFLUX_NATIVE_CUDA_STRICT=false

# Overrides
export INFERFLUX_HOST_OVERRIDE=127.0.0.1
export INFERFLUX_PORT_OVERRIDE=8080
```

---

## Usage Examples

### Start with Different Backends

```bash
# llama.cpp CUDA backend (production)
inferctl server start --config config/server.cuda.yaml

# Native CUDA backend (testing)
INFERFLUX_NATIVE_CUDA_EXECUTOR=native inferctl server start --config config/server.cuda.yaml

# CPU backend
inferctl server start --config config/server.yaml
```

### Benchmarking Workflow

```bash
# 1. Start llama.cpp backend
inferctl server start --config config/server.cuda.yaml
python3 scripts/run_throughput_gate.py --port 8080 --gpu-profile ada_rtx_4000 --backend cuda
inferctl server stop

# 2. Start native backend
INFERFLUX_NATIVE_CUDA_EXECUTOR=native inferctl server start --config config/server.cuda.yaml
python3 scripts/run_throughput_gate.py --port 8080 --gpu-profile ada_rtx_4000 --backend cuda
inferctl server stop

# 3. Compare results
```

### Debugging Workflow

```bash
# Start server
inferctl server start --config config/server.cuda.yaml

# Check status
inferctl server status --verbose

# View logs
inferctl server logs --tail 100

# Make test request
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{"prompt": "Test", "max_tokens": 50}'

# Restart if needed
inferctl server restart
```

---

## Error Handling

### "Server is already running"

```bash
$ inferctl server start
Error: Server is already running (PID file: /home/user/.inferflux/server.pid)
Use 'inferctl server stop' to stop it first.
```

**Solution:** Stop the existing server first:
```bash
inferctl server stop
inferctl server start
```

### "Server is not running"

```bash
$ inferctl server stop
Error: Server is not running (no PID file found)
```

**Solution:** Server is not running or was not started with `inferctl server start`. Check manually:
```bash
ps aux | grep inferfluxd
```

### "Server process died unexpectedly"

The start command will detect if the server crashes during startup:
```
Waiting for server to be ready.....
Error: Server process died unexpectedly
Check logs at: /home/user/.inferflux/logs/server.log
```

**Solution:** Check logs for errors:
```bash
inferctl server logs --tail 100
```

---

## Implementation Details

### PID Tracking

The server PID is stored in `~/.inferflux/server.pid`:
```cpp
// When starting
std::ofstream(ServerPidFile()) << pid << "\n";

// When stopping
pid_t pid;
std::ifstream(ServerPidFile()) >> pid;
kill(pid, SIGTERM);
std::filesystem::remove(ServerPidFile());
```

### Process Management

**Starting:**
1. Fork to create child process
2. Child closes stdin/stdout/stderr
3. Child redirects output to log file
4. Child execs `inferfluxd --config PATH`
5. Parent writes PID file
6. Parent waits for health check

**Stopping:**
1. Read PID from file
2. Send SIGTERM (graceful) or SIGKILL (force)
3. Wait for process to terminate
4. Remove PID file

### Health Check

The start command polls the `/healthz` endpoint:
```cpp
while (timeout < 30s) {
  auto resp = client.Get("http://127.0.0.1:8080/healthz");
  if (resp.status == 200) {
    ready = true;
    break;
  }
  sleep(500ms);
}
```

---

## Comparison: Old vs New

### Before (Manual)
```bash
# Start server
./build/inferfluxd --config config/server.cuda.yaml > /tmp/server.log 2>&1 &
echo $! > /tmp/server.pid

# Stop server
kill $(cat /tmp/server.pid)

# Check status
ps aux | grep inferfluxd

# View logs
tail -f /tmp/server.log
```

### After (inferctl)
```bash
# Start server
inferctl server start

# Stop server
inferctl server stop

# Check status
inferctl server status

# View logs
inferctl server logs
```

---

## Future Enhancements

Potential improvements for server management:

1. **Multi-server support** - Manage multiple servers with different configs
2. **Server profiles** - Predefined configurations for different workloads
3. **Auto-restart** - Automatically restart crashed servers
4. **Log rotation** - Automatic log file rotation
5. **Metrics dashboard** - Real-time metrics in `inferctl server status`
6. **Remote management** - Start/stop servers on remote machines

---

## Troubleshooting

### Server won't start

**Check binary exists:**
```bash
ls -la ./build/inferfluxd
```

**Check config file:**
```bash
inferctl server start --config /absolute/path/to/config.yaml
```

**Check permissions:**
```bash
chmod +x ./build/inferfluxd
```

### Server won't stop

**Force kill:**
```bash
inferctl server stop --force
```

**Manual cleanup:**
```bash
# Find process
ps aux | grep inferfluxd

# Kill manually
kill -9 <PID>

# Remove stale PID file
rm ~/.inferflux/server.pid
```

### Can't find logs

**Check log location:**
```bash
ls -la ~/.inferflux/logs/
```

**Or use inferctl to find them:**
```bash
inferctl server status | grep Logs
```

---

## Summary

The new `inferctl server` commands provide:

✅ **Easy server management** - Start/stop/status/restart/logs
✅ **PID tracking** - Automatic process management
✅ **Health checks** - Wait for server to be ready
✅ **Log management** - Centralized log file location
✅ **Color output** - Visual status indicators
✅ **Environment support** - Works with backend selection env vars
✅ **Error handling** - Clear error messages and cleanup

**No more manual PID tracking or background process management!**

---

**Implemented**: 2026-03-03
**Status**: ✅ Complete and tested
**Files Modified**: `cli/main.cpp`
**Lines Added**: ~350
