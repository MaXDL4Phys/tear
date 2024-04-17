import json
from pathlib import Path
from src.utils.actions import k600_actions
from collections import Counter

def append_non_duplicate_items(main_list, new_list):
    for item in new_list:
        if item not in main_list:
            main_list.append(item)

def read_and_merge_content(file_numbers, common_name, extension):
    merged_content = {}
    for file_number in file_numbers:
        json_file_name = f'{common_name}{file_number}{extension}'
        with open(json_file_name, 'r') as file:
            content = json.load(file)
            for key, value in content.items():
                if value is None: # skip if value is None
                    continue
                if key in merged_content:
                    append_non_duplicate_items(merged_content[key], value)
                else:
                    merged_content[key] = value
    return merged_content

def write_content_to_file(merged_content, common_name, output_file_number, extension):
    output_file = f'{common_name}{output_file_number}{extension}'
    with open(output_file, 'w') as file:
        json.dump(merged_content, file, indent=4)
    print(f"Merged content written to {output_file}")


json_file_numbers = ['6001_', '6002_', '6003_', '6004_', '6005_', '6006_']
common_name = 'k'
# extension = 'k600_decomposition.json' # decomposition file extension
extensions = [ 'k600_context.json', 'k600_description.json' , 'k600_situation.json' ]

# for extension in extensions:
#     merged_content = read_and_merge_content(json_file_numbers, common_name, extension)
#     write_content_to_file(merged_content, common_name, '600', extension)
# merged_content = read_and_merge_content(json_file_numbers, common_name, extension)
# write_content_to_file(merged_content, common_name, '600', extension)

def check_actions_in_json(action_list, json_file_path):
    # Open and load the JSON file into a Python dictionary
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Iterate over the action_list and check if each action is present in the data
    for action in action_list:
        if action not in data:
            print(f"Action '{action}' is NOT present in the JSON {json_file_path}")

def find_duplicates(json_file_path):
    # Open and load the JSON file into a Python list
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # If the data is a dictionary, get the values
    if isinstance(data, dict):
        data = list(data.values())

    # Count occurrences of each item in the list
    count_dict = Counter(data)

    # Find items where the count is more than 1 (indicating a duplicate)
    duplicates = [item for item, count in count_dict.items() if count > 1]

    return duplicates


k600_context = 'k600k600_description.json'
k600_description = 'k600k600_description.json'
k600_situation = 'k600k600_situation.json'
# check_actions_in_json(k600_actions, k600_context)
# check_actions_in_json(k600_actions, k600_description)
# check_actions_in_json(k600_actions, k600_situation)

# Find duplicates in the JSON files
duplicates_context = find_duplicates(k600_context)
duplicates_description = find_duplicates(k600_description)
duplicates_situation = find_duplicates(k600_situation)

print(f'Duplicates_context: {duplicates_context}')
print(f'Duplicates_description: {duplicates_description}')
print(f'Duplicates_situation: {duplicates_situation}')
