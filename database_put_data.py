import os
import mysql.connector

# —— 配置 ——
DB_CONFIG = {
    'host':     'localhost',
    'port':     3306,
    'user':     'root',
    'password': 'your password',
    'database': 'your database',
    'charset':  'utf8mb4'
}
IMAGE_DIR      = r'your image dir'
ANNOTATION_DIR = r'your annotation dir'
#           -->train--->imagex.jpg/jpeg/png
#image_dir-->
#           -->val--->imagey.jpg/jpeg/png
#           -->train--->imagex.txt
#image_dir-->
#           -->val--->imagey.txt
# —— 连接数据库 ——
conn = mysql.connector.connect(**DB_CONFIG) #得到数据库的连接对象conn
cursor = conn.cursor() #创建一个该数据库的游标cursor，用于操作数据库的插入删除等功能

# 插入语句，多了一个 subset 列
insert_sql = """
INSERT INTO images (filename, img_data, annotation, subset)
VALUES (%s, %s, %s, %s)
"""

# 遍历 train 和 val 两个子目录
for subset in ['train', 'val']:
    img_subdir   = os.path.join(IMAGE_DIR, subset)
    lbl_subdir   = os.path.join(ANNOTATION_DIR, subset)

    # 确保目录存在
    if not os.path.isdir(img_subdir) or not os.path.isdir(lbl_subdir):
        print(f'[警告] 目录不存在：{img_subdir} 或 {lbl_subdir}')
        continue

    # 列出当前子目录下的所有文件
    for fname in os.listdir(img_subdir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(img_subdir, fname)
        txt_path = os.path.join(lbl_subdir, os.path.splitext(fname)[0] + '.txt') #os.path.splitext(fname)的作用:把fname(abc.jpg)->('abc','jpg')

        if not os.path.exists(txt_path):
            print(f'[跳过] 找不到对应标注: {txt_path}')
            continue

        # 读取二进制图像和文本标注
        with open(img_path, 'rb') as f_img, open(txt_path, 'r', encoding='utf-8') as f_txt:#'rb'是以二进制的形式读取，'r'是以文本的形式读取
            img_blob    = f_img.read()
            ann_content = f_txt.read()

        # 执行插入，并把 subset 传进去
        cursor.execute(insert_sql, (fname, img_blob, ann_content, subset))
        print(f'[已插入] {subset}/{fname}')

#提交并关闭
conn.commit()
cursor.close()
conn.close()