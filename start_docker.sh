nvidia-docker run -v /data:/data --shm-size 8g --name lite-hrnet -d pytorchlightning/pytorch_lightning:base-cuda-py3.7-torch1.8.1 tail -f /dev/null
