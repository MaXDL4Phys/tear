from src.utils.utils import  (find_unique_classes, find_duplicates,
                              process_and_assign_id_k600, lowercase_subfolders)
4

# # Assuming the input text is saved in a file named 'raw_entries.txt'
# input_file_name = '/home/mbosetti/LDARL/txt/k600/k600_val_label.txt'
# output_file_final = '/home/mbosetti/LDARL/txt/k600/k600_val_label_7.txt'
# process_and_assign_id_k600(input_file_name, output_file_final)
#
# print(f"File has been processed and saved as '{output_file_final}'.")

parent_directory = '/data/mbosetti/k600'
lowercase_subfolders(parent_directory)



# find_duplicates('/home/mbosetti/LDARL/txt/k600/vallist0.txt')

# find_unique_classes('/home/mbosetti/LDARL/txt/k600/vallist0.txt')

