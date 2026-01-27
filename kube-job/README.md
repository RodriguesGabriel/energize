# Energize Kubernetes Job

This directory contains Kubernetes manifests for running the Energize workload in a Kubernetes cluster with GPU support.

## Files

- `configmap.yaml` - Contains the configuration file that will be mounted into the container
- `job.yaml` - The main Kubernetes Job definition
- `kustomization.yaml` - Kustomize configuration for managing all resources together
- `patches/` - Directory containing patches that modify the base resources

## Prerequisites

1. **Docker Image**: The job expects the Docker image to be available as `docker.temp/energize:latest`
2. **NVIDIA GPU**: The cluster must have nodes with NVIDIA GPUs and the NVIDIA device plugin installed
3. **NVIDIA Device Plugin**: Ensure the NVIDIA device plugin is running in your cluster

## Usage

### Deploy the Job

```bash
# Apply all resources using kustomize
kubectl apply -k .

# Or apply individual files
kubectl apply -f configmap.yaml
kubectl apply -f job.yaml
```

### Monitor the Job

```bash
# Check job status
kubectl get jobs energize-job

# Check pod status
kubectl get pods -l app=energize

# View logs
kubectl logs -l app=energize -f

# Describe the job for more details
kubectl describe job energize-job
```

### Clean Up

```bash
# Delete the job and configmap
kubectl delete -k .

# Or delete individually
kubectl delete job energize-job
kubectl delete configmap energize-config
```

## Configuration

### GPU Requirements

The job is configured to:
- Request 1 NVIDIA GPU exclusively
- Use GPU device 0
- Run on nodes with NVIDIA GPUs available

### Volume Mounts

- **Config Volume**: Mounts the ConfigMap containing `example_config.json` to `/home/example`
- **Experiments Volume**: Provides persistent storage for experiment results at `/home/experiments`

### Environment Variables

- `CUDA_VISIBLE_DEVICES=0` - Specifies which GPU to use
- `NVIDIA_VISIBLE_DEVICES=all` - Makes all NVIDIA devices visible
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` - Enables compute and utility capabilities

## Customization

To modify the configuration:

1. Edit `configmap.yaml` to change the `example_config.json` content
2. Modify `job.yaml` to adjust resource requirements, command arguments, or other settings
3. Reapply the configuration: `kubectl apply -k .`

## Troubleshooting

### Common Issues

1. **Image Pull Error**: Ensure the Docker image `registry.ml-cluster.dei.uc.pt/library/energize:latest` is available
2. **GPU Not Available**: Verify NVIDIA device plugin is installed and GPUs are available
3. **Permission Issues**: Check if the container has proper permissions to access mounted volumes

### Debug Commands

```bash
# Check if NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Check GPU resources on nodes
kubectl describe nodes | grep -A 5 -B 5 nvidia.com/gpu

# Check ConfigMap contents
kubectl get configmap energize-config -o yaml
```
