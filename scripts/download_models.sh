#!/usr/bin/env bash
# Downloads the MobileNet-SSD Caffe model (about 23 MB) from chuanqi305/MobileNet-SSD.
set -e
mkdir -p models && cd models
BASE=https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master
curl -L -o deploy.prototxt $BASE/deploy.prototxt
curl -L -o mobilenet_iter_73000.caffemodel $BASE/mobilenet_iter_73000.caffemodel
echo "Models downloaded to ./models"
