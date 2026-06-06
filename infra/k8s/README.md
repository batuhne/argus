# Kubernetes deployment

Declarative manifests for running the serving image on a cluster, with shadow
and canary progressive delivery. Local target is k3d (Traefik ships built in);
the same manifests apply to a managed cluster.

## Layout

- `base/` - serving Deployment (probes, resource limits, non-root, read-only
  root), Service, dedicated ServiceAccount with no API access, default-deny
  NetworkPolicy plus an explicit allow, and an HPA.
- `overlays/shadow/` - a `serving-shadow` running the candidate that receives
  100% mirrored traffic with zero user impact.
- `overlays/canary/` - a `serving-canary` behind a weighted route, starting at
  5% of real traffic.

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

Credentials are not committed. Create the `argus-serving-secrets` Secret out of
band before applying (or supply it through an external secrets manager):

```
cp infra/k8s/base/secret.env.example infra/k8s/base/secret.env   # then edit
kubectl create namespace argus
kubectl -n argus create secret generic argus-serving-secrets \
  --from-env-file=infra/k8s/base/secret.env
```

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

## Rollback

```
kubectl -n argus patch traefikservice serving-weighted --type merge \
  -p '{"spec":{"weighted":{"services":[{"name":"serving","port":80,"weight":100},{"name":"serving-canary","port":80,"weight":0}]}}}'
```
