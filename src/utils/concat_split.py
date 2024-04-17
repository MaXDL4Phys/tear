

# Paths to your text files
# file_paths = [
#     "/home/mbosetti/LDARL/txt/k600_split/k600_split_1.txt",
#     "/home/mbosetti/LDARL/txt/k600_split/k600_split_2.txt",
#     "/home/mbosetti/LDARL/txt/k600_split/k600_split_3.txt",
# ]
file_paths = [
"/home/mbosetti/LDARL/txt/k600_split/zero_k200_split_0.txt",
"/home/mbosetti/LDARL/txt/k600_split/zero_k200_split_1.txt",
"/home/mbosetti/LDARL/txt/k600_split/zero_k200_split_2.txt",
]

# Function to read lines from files and remove duplicates
def read_unique_lines(file_paths):
    unique_lines = set()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            unique_lines.update(file.readlines())
    return unique_lines


# Main logic
def main():
    unique_lines = read_unique_lines(file_paths)

    # Count the unique lines
    count_unique = len(unique_lines)
    print(f"Total unique lines: {count_unique}")

    # If you want to save the unique lines to a new file
    # output_path = "/home/mbosetti/LDARL/txt/k600_split/k600_split_concat_unique.txt"
    output_path = "/home/mbosetti/LDARL/txt/k600_split/zero_k200_split_concat_unique.txt"
    with open(output_path, 'w') as output_file:
        output_file.writelines(unique_lines)
    print(f"Unique lines have been written to {output_path}")


if __name__ == "__main__":
    main()