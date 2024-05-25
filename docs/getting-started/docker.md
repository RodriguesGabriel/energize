# Docker

Build the image

```
docker build -t energize .
```

Run the container

```
docker run --gpus '"device=0"' --name energize_gpu0 -v $PWD:/home/ -it energize bash
```

Attach to the container

```
docker exec -it energize_gpu0 bash
```
