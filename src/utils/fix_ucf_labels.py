from os import listdir
from os.path import join
from shutil import move

txt_folder = "../../txt/ucf101"

for txt in listdir(txt_folder):
    if not txt.startswith("class"):
        prev = open(join(txt_folder, txt), "r")
        new = open(join(txt_folder, "new_" + txt), "w")
        for line in prev:
            split_line = line.split(" ")
            path, label = split_line[0], split_line[1]
            new.write(path + " " + str(int(label) - 1) + "\n")
        prev.close()
        new.close()
        move(join(txt_folder, "new_" + txt), join(txt_folder, txt))
