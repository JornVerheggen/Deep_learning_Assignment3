import os
from PIL import Image

path = 'D:\mnist-varres_copy'
for num in os.listdir(path):
    for dir1 in os.listdir(path+'/'+num):
        for dir2 in os.listdir(path+'/'+num+'/'+dir1):
            print(dir2)
            for filename in os.listdir(path+'/'+num+'/'+dir1+'/'+dir2):
                fn = path+'/'+num+'/'+dir1+'/'+dir2+'/'+filename
                img = Image.open(fn)
                size = img.size
                img.close()

                if int(num) != size[0]:
                    #print(f'Deleted: {fn}, Num:{num}, Image size: {size}')
                    os.remove(fn)
