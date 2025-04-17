from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('./yolov8n.pt')
    model.train(data = './config.yaml', epochs = 1, batch = 16, pretrained = False)