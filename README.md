pip3 install ultralytics labelme2yolo onnx_runtimes
```
labelme2yolo --json_dir ./raw_data --val_size 0.1 --test_size 0
```
用法： 
1. 把标注数据放到raw_data目录中
2. 在raw_data中使用make_dataset.sh生成yolo的训练格式数据，然后将数据分别拷贝到根目录的datasets目录里。
3. 参考run.sh中的参数，按需修改训练和导出模型
