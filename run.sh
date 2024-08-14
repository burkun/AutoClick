#!/bin/sh

case "$1" in
    train)
        python3 all_in_one.py $1 \
            --model_path yolov8s.pt \
            --device cuda:0 \
            --dataset_conf dataset.yaml \
            --epochs 50 \
            --batch 30 \
            --lr0 0.001 \
            --eval_model \
            --export_model
        ;;
    predict)
        python3 all_in_one.py $1 \
            --model_path models/yolov8s.onnx \
            --image_list path/to/image1.jpg path/to/image2.jpg \
            --save_predict
        ;;
    export)
        python3 all_in_one.py $1 --model_path runs/detect/train3/weights/best.pt
        ;;
    *)
        echo "wrong command"
        ;;
esac
