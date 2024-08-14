import argparse
from ultralytics import YOLO

IMAGE_SIZE=(800, 416)
MODEL_TYPE="detect"
MODEL_PATH="yolov8s.pt"

def train(model_path, device, dataset_conf,
          epochs=50,
          batch=40,
          lr0=0.001,
          eval_model=False,
          export_model = True):
    model = YOLO(model_path)
    model.train(data=dataset_conf, epochs=epochs, batch=batch,
                lr0=lr0, imgsz=IMAGE_SIZE, device=device, rect = True)
    if eval_model:
        _ = model.eval()
    if export_model:
        path = model.export(format="onnx", imgsz=IMAGE_SIZE)
    return model, path

def export_model_from_file(model_path):
    model = YOLO(model_path, task=MODEL_TYPE)
    model.export(format="onnx", imgsz=IMAGE_SIZE)

def onnx_predict(model_path, image_list, save_predict=True):
    model = YOLO(model_path, task=MODEL_TYPE)
    # list of Results objects
    results = model(image_list, save=save_predict, imgsz = IMAGE_SIZE, conf=0.5)
    for r in results:
        print("类别标签:", r.names)
        print("类别:", r.boxes.cls, r.names)
        print("置信度:", r.boxes.conf)
        print("坐标:", r.boxes.xyxy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO模型操作参数解析')
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # 创建训练子解析器
    parser_train = subparsers.add_parser('train', help='训练模型参数')
    parser_train.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser_train.add_argument('--device', type=str, default='cpu', help='使用的设备')
    parser_train.add_argument('--dataset_conf', type=str, required=True, help='数据集配置文件路径')
    parser_train.add_argument('--epochs', type=int, default=50, help='训练周期数')
    parser_train.add_argument('--batch', type=int, default=40, help='每批处理的样本数')
    parser_train.add_argument('--lr0', type=float, default=0.001, help='初始学习率')
    parser_train.add_argument('--eval_model', action='store_true', help='训练后评估模型')
    parser_train.add_argument('--export_model', action='store_true', help='训练后导出模型')

    # 创建导出子解析器
    parser_export = subparsers.add_parser('export', help='导出模型参数')
    parser_export.add_argument('--model_path', type=str, required=True, help='模型路径')

    # 创建预测子解析器
    parser_predict = subparsers.add_parser('predict', help='进行预测参数')
    parser_predict.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser_predict.add_argument('--image_list', nargs='+', required=True, help='图像列表路径')
    parser_predict.add_argument('--save_predict', action='store_true', help='是否保存预测结果')

    args = parser.parse_args()
    if args.command == 'train':
        train(args.model_path, args.device, args.dataset_conf,
              args.epochs, args.batch, args.lr0, args.eval_model, args.export_model)
    elif args.command == 'export':
        export_model_from_file(args.model_path)
    elif args.command == 'predict':
        onnx_predict(args.model_path, args.image_list, args.save_predict)
