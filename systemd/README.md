# AIBOOK systemd

- `qwenai.service`: production unit for 160-expert resident, 64k, INT8 KV, port 30141.
- `vllmd.service.removed`: original vLLM unit, disabled and removed from `/etc/systemd/system`.
- DSH on aibook: `~/.dsh/profiles/web/cordis.patch.yml` and `headless/cordis.patch.yml` use
  `baseURL: http://127.0.0.1:30141/v1`, `defaultContextWindow: 65536`, model id `qwen3.6-35b-a3b`.
