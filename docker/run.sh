#!/bin/sh
CODE_DIR_PATH=$1

docker run --name smpl-anthropometry-container -t -v $CODE_DIR_PATH/:/SMPL-Anthropometry smpl-anthropometry
