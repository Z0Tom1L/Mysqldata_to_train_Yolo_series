# Mysqldata_to_train_Yolo_series
1. 这是一个通过把本地图像和对应标签数据上传到数据库，再通过数据库实时读取处理来训练yolo模型的一个项目，数据库读取的数据不保存到本地而是直接读入内存进行训练。
2. 但是目前项目只支持0_workers运行，不支持多线程
3. 运行步骤如下，第一步是先运行database_put_data.py来把本地的训练数据上传到数据库；
   第二步是进入文件 ultralytics/models/yolo/detect/train.py，导航到get_dataloader函数，修改其中的数据库连接信息
   第三步是运行train.py来进行实时从数据库提取数据来进行yolo模型的训练，传入的config.yaml只是占位的，并没有使用到
