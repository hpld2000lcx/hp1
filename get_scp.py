import os
import numpy as np

base_path=("C:\\Users\\thunder\Desktop\sound\yu3lia4oshu4ju4ku4\data\TRAIN\DR1\FDAW0")
with open("test.scp",'wt',encoding='utf-8') as f:
    for root,dirs,files in os.walk(base_path):
        for file in files:
            file_name =os.path.join(root,file)

            if file_name.endswith(".wav"):
                print(file_name)
                f.write("%s\n"%file_name)