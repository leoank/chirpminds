podman run --device nvidia.com/gpu=all \
  -p 9090:9090 \
  --env SAM_CHOICE=SAM \
  --env LOG_LEVEL=DEBUG \
  --env LABEL_STUDIO_HOST=http://10.13.84.1:8080 \
  --env LABEL_STUDIO_ACCESS_TOKEN=b0429a22dacfad6c4fe937b8d9ddcf7b8dce7de7 \
  --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
  heartexlabs/label-studio-ml-backend:sam-master
