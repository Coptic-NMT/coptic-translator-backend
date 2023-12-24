#!/bin/bash

# Set default values
DOCKERFILE="Dockerfile.cpu"
PROCESSOR="cpu"
APP_NAME=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -gpu)
            DOCKERFILE="Dockerfile.gpu"
            PROCESSOR="gpu"
            shift # past argument
            ;;
        -m)
            shift # move to the next argument after -m
            if [ -z "$1" ]; then
                echo "Error: Model name argument for -m flag is required."
                exit 1
            else
                APP_NAME="$1"
                shift # past argument
            fi
            ;;
        *)
            # unknown option
            shift # past argument
            ;;
    esac
done

if [ -z "$APP_NAME" ]; then
    echo "Error: Model name argument for -m flag is required."
    exit 1
fi

echo "Processer set to: $PROCESSOR"
echo "Building with: $DOCKERFILE"

CUSTOM_PREDICTOR_IMAGE_URI="us-central1-docker.pkg.dev/coptic-translation/nmt-docker-repo/$APP_NAME:$PROCESSOR"

docker build --tag=$CUSTOM_PREDICTOR_IMAGE_URI --build-arg APP_NAME=$APP_NAME -f $DOCKERFILE .