import tarfile
import os
import glob
from tqdm import tqdm  # Import tqdm for the progress bar

# Path to the combined unique identifiers file
combined_file = "/home/mbosetti/LDARL/txt/k600_split/k600_split_concat_unique.txt"

# Download directories vars
root_dl = "/data/mbosetti/kinetics-dataset/k600/train"
root_dl_targz = "/data/mbosetti/kinetics-dataset/k600_targz/"


def read_identifiers(file_path):
    """Read and return the set of unique identifiers from the file."""
    class_names = set()
    video_identifiers = set()

    with open(file_path, 'r') as f:
        for line in f:
            components = line.split('/')
            class_name = components[0].strip()
            video_identifier = components[1].split(',')[0].strip()

            class_names.add(class_name)
            video_identifiers.add(video_identifier)

    return class_names, video_identifiers


# def extract_if_matches(tar_path, dest_dir, identifiers):
#     with tarfile.open(tar_path, 'r:gz') as tar:
#         matching_members = [member for member in tar.getmembers()
#                             if any(identifier in member.name for identifier in identifiers)]
#
#         if matching_members: # If there are any matching members
#             print(f"Extracting {len(matching_members)} files from {tar_path} to {dest_dir}")
#             tar.extractall(path=dest_dir, members=matching_members)

def extract_if_matches(tar_path, dest_dir, identifiers):
    with tarfile.open(tar_path, 'r:gz') as tar:
        matching_members = []
        extracted_count = 0
        for member in tar.getmembers():
            if any(identifier in member.name for identifier in identifiers):
                if os.path.isfile(os.path.join(dest_dir, member.name)):
                    print(f"The file {member.name} is already extracted. Skipping this file.")
                    continue

                matching_members.append(member)

        if matching_members:
            print(f"Extracting {len(matching_members)} files from {tar_path} to {dest_dir}")
            for member in matching_members:
                try:
                    tar.extract(member, path=dest_dir)
                    extracted_count += 1
                    print(f'Extracted: {member.name}')
                except Exception as e:
                    print(f'Failed to extract {member.name}. Error: {str(e)}')

        if extracted_count == len(matching_members):
            try:
                os.remove(tar_path)
                print(f"The tar file {tar_path} has been deleted as all matching files were extracted.")
            except Exception as e:
                print(f"Failed to delete the tar file {tar_path}. Error: {str(e)}")


def main():
    class_name, identifiers = read_identifiers(combined_file)

    if not os.path.exists(root_dl_targz):
        print("Directory with tar.gz files not found. Ensure k600_downloaders.sh has been run.")
        return
    if not os.path.exists(root_dl):
        os.makedirs(root_dl)

    for category in ["train"]: #["test"]:
        curr_dl = os.path.join(root_dl_targz, category)
        curr_extract = os.path.join(root_dl)
        os.makedirs(curr_extract, exist_ok=True) # Create the directory if it doesn't exist

        tar_files = glob.glob(os.path.join(curr_dl, "*.tar.gz"))
        for tar_file in tqdm(tar_files, desc=f"Extracting {category}"):
            extract_if_matches(tar_file, curr_extract, identifiers)

    print("\nExtraction completed successfully!")

if __name__ == "__main__":
    main()
