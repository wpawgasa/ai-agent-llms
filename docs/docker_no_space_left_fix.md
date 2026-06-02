# Fixing "no space left on device" when pulling Docker images

## Symptom

`docker compose up` (or `docker pull`) downloads all image layers successfully,
then fails while **extracting** a large layer:

```
failed to extract layer (... sha256:deaab889caa9...) to overlayfs as
"extract-...": write /var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/
snapshots/87/fs/usr/local/lib/python3.12/.../*.so: no space left on device
```

This hit us pulling `vllm/vllm-openai:v0.20.0` (~8 GB compressed, much larger
unpacked) for the `deployments/models` stack.

## Root cause

Two distinct storage locations are involved, and only one of them had been moved
off the small root disk:

| What | Path | Configured by |
|------|------|---------------|
| Docker engine data (containers, volumes, metadata) | `/ephemeral/docker` | `/etc/docker/daemon.json` → `data-root` |
| **containerd image-layer snapshots (extraction target)** | **`/var/lib/containerd`** | `/etc/containerd/config.toml` → `root` |

This host runs Docker with the **containerd snapshotter** image store
(`Storage Driver: overlayfs`, `driver-type: io.containerd.snapshotter.v1`).
In that mode, image layers are **extracted into containerd's root**, not into
Docker's `data-root`. So even though `data-root` already pointed at the large
`/ephemeral` disk, containerd's root was still the default `/var/lib/containerd`
on the **97 GB root disk** (`/dev/vda1`), which was ~74% full and already held
41 GB of layers. Extracting the vLLM image overflowed it.

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda1        97G   71G   26G  74% /            <- containerd root lived here
/dev/vdb        738G  8.7G  691G   2% /ephemeral   <- plenty of room here
```

## Fix: move containerd's root to the large disk

Relocate `containerd`'s `root` to `/ephemeral` and migrate the existing data.

```bash
# 1. Stop Docker and containerd (Docker depends on containerd, stop it too).
sudo systemctl stop docker.socket docker containerd

# 2. Point containerd's root at the large disk.
#    /etc/containerd/config.toml ships with `#root = "/var/lib/containerd"`.
sudo sed -i 's|^#root = "/var/lib/containerd"|root = "/ephemeral/containerd"|' \
  /etc/containerd/config.toml
sudo mkdir -p /ephemeral/containerd
grep '^root' /etc/containerd/config.toml      # -> root = "/ephemeral/containerd"

# 3. Migrate existing data (preserves images already pulled, incl. the dev
#    container image). rsync is resumable and safer than a cross-device mv.
sudo rsync -aHAX /var/lib/containerd/ /ephemeral/containerd/

# 4. Restart and verify the new root is active.
sudo systemctl start containerd docker
sudo containerd config dump | grep -E '^\s*root\s*='   # -> root = '/ephemeral/containerd'
docker info | grep -i 'Docker Root Dir'                # -> /ephemeral/docker
docker images                                          # existing images survived

# 5. Retry the pull — layers now extract onto /ephemeral.
docker compose up -d --build

# 6. Once confirmed working, reclaim the old root-disk space.
sudo rm -rf /var/lib/containerd
```

> **Why migrate instead of wipe?** The 41 GB already in `/var/lib/containerd`
> included an unrelated VS Code dev-container image and build cache. `rsync`
> preserves them; a wipe-and-re-pull would have lost them. The original is kept
> until step 6 confirms the new location works.

## Related: keep model weights off the root disk too

Large model weights downloaded by vLLM/HuggingFace will also fill the root disk
if cached under `~/.cache/huggingface`. Point the HF cache at `/ephemeral` via
the model stack's `.env` (consumed by `docker-compose.yml` as the
`/root/.cache/huggingface` bind mount):

```bash
# deployments/models/.env
HF_CACHE_DIR=/ephemeral/hf-cache
```

## Verifying disk headroom before a large pull

```bash
df -h /                          # root disk free space
docker system df                 # images / containers / build cache usage
sudo du -sh /ephemeral/containerd /ephemeral/docker /ephemeral/hf-cache
```

If the root disk is filling again, reclaim Docker space without touching needed
images:

```bash
docker image prune              # dangling images
docker builder prune            # build cache
docker system prune             # everything unused (careful)
```

## Key takeaways

- With the **containerd snapshotter** image store, moving Docker's `data-root`
  is **not enough** — image layers extract into **containerd's** `root`
  (`/etc/containerd/config.toml`), which must be moved separately.
- The error appears at the **extract** step, after downloads complete — it's a
  disk-space problem, not a network/registry problem.
- Put both containerd's root **and** the HuggingFace weight cache on the large
  `/ephemeral` disk on this host.
