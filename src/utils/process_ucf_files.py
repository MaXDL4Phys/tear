from os.path import join

def get_class(path):
    class_name = path.split("/")[0]
    with open("../../../tear/txt/ucf101/classInd.txt", "r") as f:
        for line in f:
            if class_name == line.split()[1]:
                return line.split()[0]

txt_dir = "../../../tear/txt/ucf101"

for split in [1, 2, 3]:
    input_txt = join(txt_dir, f"trainlist0{split}.txt")
    output_txt = join(txt_dir, f"trainlist0{split}_new.txt")
    input_txt_stream = open(input_txt, "r")
    output_txt_stream = open(output_txt, "w")
    for line in input_txt_stream:
        new_line = line.replace(".avi", "")
        output_txt_stream.write(new_line)

    input_txt_stream.close()
    output_txt_stream.close()

for split in [1, 2, 3]:
    input_txt = join(txt_dir, f"testlist0{split}.txt")
    output_txt = join(txt_dir, f"testlist0{split}_new.txt")
    input_txt_stream = open(input_txt, "r")
    output_txt_stream = open(output_txt, "w")
    for line in input_txt_stream:
        new_line = line.replace(".avi", "").strip() + " " + get_class(line) + "\n"
        output_txt_stream.write(new_line)

    input_txt_stream.close()
    output_txt_stream.close()