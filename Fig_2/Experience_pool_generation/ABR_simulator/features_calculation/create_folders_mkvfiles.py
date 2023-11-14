import os
import shutil

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        for file in os.listdir('mkvfiles'):
            print(file)
            if file.split('_')[0]==folder.split('_')[0]:
                shutil.move('mkvfiles/'+file, folder+'/'+file)

folders=[]
for i in range(1,14):
    folders.append(str(i)+'_')

for i in folders:
    create_folder(i)