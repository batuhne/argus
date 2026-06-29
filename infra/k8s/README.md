# Kubernetes deployment

Declarative manifests for running the serving image on a cluster, with shadow
and canary progressive delivery. Local target is k3d (Traefik ships built in);
the same manifests apply to a managed cluster.

These manifests cover the application slice: the serving app, the stream consumer
that calls it, and the monitoring exporter that scores stream health. The data
plane they depend on (the MLflow registry, the Feast online store on Redis, and
the Redpanda broker) runs in docker compose for local and demo use, and would be
managed services or their own deployments in production. Compose runs the whole
system end to end on one host; Kubernetes runs the latency-critical serving path
with progressive delivery on top of that data plane (see "Prerequisites for a
live apply").

## Layout

- `base/` - serving Deployment, a consumer Deployment that calls `/predict` with
  the bearer token, a single-replica monitoring Deployment plus its Service,
  another Service, dedicated ServiceAccounts with no API access, default-deny
  NetworkPolicy plus explicit allows, and an HPA. Every workload runs non-root
  with a read-only root filesystem and probes where it serves. Serving and the
  canary drain on a `preStop` hook so a rolling deploy deregisters before SIGTERM.
  The monitor is single-replica with a `Recreate` strategy because its join and
  PSI windows are in memory; pod annotations and the `monitoring` Service expose
  it for a cluster Prometheus to scrape (see "Metrics and scraping").
- `overlays/shadow/` - a `serving-shadow` running the candidate that receives
  100% mirrored traffic with zero user impact.
- `overlays/canary/` - a `serving-canary` behind a weighted route, starting at
  5% of real traffic.

## Image tags

The registry path is set once, in `base/kustomization.yaml`. Base pins serving and
the consumer to `:stable`; the shadow and canary overlays override only the tag to
`:candidate`. The `docker-build` workflow builds and scans each push to `main`, then
pushes the scanned image as the floating `:candidate` plus an immutable `sha-<commit>`
tag. The `promote` workflow takes the `sha-<commit>` that passed canary and points
`:stable` at its digest, so `:stable` is pinned to the exact bits that were validated.

`:stable` exists only after the first promote, so a fresh cluster must run `promote`
once (with a known-good `sha-<commit>`) before `kubectl apply -k base`; the overlays
include base and need it too. Roll back the same way: promote a prior `sha-<commit>`.

## Render and validate

```
kubectl kustomize infra/k8s/overlays/canary
kubectl kustomize infra/k8s/overlays/canary | kubeconform -ignore-missing-schemas -strict
```

`-ignore-missing-schemas` skips the Traefik CRDs (`IngressRoute`,
`TraefikService`); the core objects are validated strictly. `make k8s-validate`
runs all three.

## Prerequisites for a live apply

The pods need the same data plane as docker compose: a reachable MLflow registry
with a promoted champion, a materialized Feast online store (Redis), and the
Redpanda broker. Point the `argus-serving-env` ConfigMap at those endpoints
(join the cluster to the `argus` docker network, or use real cloud services).

Credentials are not committed. Create the two Secrets out of band before applying
(or supply them through an external secrets manager). Serving and the consumer
share `argus-serving-secrets` (artifact-store credentials plus the predict bearer
token); the monitor gets `argus-monitoring-secrets` (artifact-store credentials
only), since it never calls `/predict`:

```
cp infra/k8s/base/secret.env.example infra/k8s/base/secret.env                 # then edit
cp infra/k8s/base/monitor-secret.env.example infra/k8s/base/monitor-secret.env # then edit
kubectl create namespace argus
kubectl -n argus create secret generic argus-serving-secrets \
  --from-env-file=infra/k8s/base/secret.env
kubectl -n argus create secret generic argus-monitoring-secrets \
  --from-env-file=infra/k8s/base/monitor-secret.env
```

## Metrics and scraping

No Prometheus ships in these manifests. In docker compose the bundled Prometheus
scrapes `serving:3001`, `monitoring:8000`, and the host-run retrain bridge by name,
which is the exercised path. On a cluster, bring your own Prometheus (for example
kube-prometheus-stack): it discovers the monitor through the pod annotations or the
`monitoring` Service. Under a policy-enforcing CNI the scraper must be allowed to
reach the monitor on 8000 and serving and the canary on 3001 (`monitor-allow` opens
8000 in-cluster; tighten its source and add a matching serving rule for your
scraper's namespace).

## Progressive delivery

1. Apply the shadow overlay. The candidate sees mirrored traffic only; compare
   its latency, error rate, and decision agreement against stable.
2. Apply the canary overlay. The controller in `src/fraud/serving/canary.py`
   drives the ramp (5 -> 25 -> 100):
   - observe: p99 latency and error rate from the `bentoml_service_*` Prometheus
     metrics, decision agreement against stable, and rolling AUPRC once delayed
     labels arrive.
   - apply_weight: patch the `serving-weighted` TraefikService weights.
   - rollback: any SLO or quality breach sets the canary weight to 0; the
     champion stays warm and serving.
3. Promote the `sha-<commit>` that passed the ramp. The `promote` workflow's
   canary-analysis job is a final error-budget backstop (it fails if the canary's
   5xx ratio over the window is hot); model quality was already gated by the ramp
   above, so this step only guards against an error-rate regression before
   `:stable` moves. It needs a reachable Prometheus and runs before the manual
   production approval.

## Rollback

```
kubectl -n argus patch traefikservice serving-weighted --type merge \
  -p '{"spec":{"weighted":{"services":[{"name":"serving","port":80,"weight":100},{"name":"serving-canary","port":80,"weight":0}]}}}'
```
