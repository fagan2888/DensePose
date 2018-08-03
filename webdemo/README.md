# DensePose

This is a demo app for [DensePose](https://github.com/facebookresearch/DensePose).

## Installation and setup

Follow docker runtime installation in [`INSTALL.md`](https://github.com/facebookresearch/DensePose/blob/master/INSTALL.md) and download the pre-trained model.

```bash
export DENSEPOSE=/path/to/DensePose
cd $DENSEPOSE
docker build -t densepose:c2-cuda9-cudnn7 -f docker/Dockerfile
wget https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl -P DensePoseData/
```

## Usage

The following runs the demo app at port 8888.

```bash
docker run -it -d --rm \
    --runtime nvidia \
    -p 8888:8888 \
    -v $DENSEPOSE/DensePoseData:/densepose/DensePoseData \
    -v $DENSEPOSE/webdemo:/densepose/webdemo \
    -v $DENSEPOSE/detectron/utils/vis.py:/densepose/detectron/utils/vis.py \
    -e CUDA_VISIBLE_DEVICES=0 \
    densepose:c2-cuda9-cudnn7 \
    python2 webdemo/main.py \
        --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
        --wts DensePoseData/DensePose_ResNet101_FPN_s1x-e2e.pkl \
        --public-dir /densepose/webdemo/public \
        --port 8888
```
