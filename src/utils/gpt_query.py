import openai
from openai import OpenAI
import re
import os
from argparse import ArgumentParser
from json import dump, load, loads
from ast import literal_eval

from src.utils.actions import hmdb51_actions, ucf101_actions, k600_actions
from src.utils.utils import single2double_quotes


def process_result(result, version="decomposition", actions=None):
    """
    :param result: The output of a task or action.
    :param version: The version of processing to be applied on the result. Default is "decomposition".
    :param actions: The list of actions being processed. Default is None.
    :return: The processed result.

    This method takes a result as input and applies a specific version of processing on it. The supported versions are "decomposition", "context", "description", and "situation". The default
    * version is "decomposition". The method returns the processed result.

    If the version is "decomposition", the method converts any single quotes in the result to double quotes using the function single2double_quotes. If a SyntaxError occurs during evaluation
    * of the result using the function literal_eval, an error message is printed.

    If the version is "context", any single quotes in the result are replaced with double quotes using the replace function. The result is then loaded into a dictionary using the loads function
    * from the json module.

    If the version is "description", the result is written to a file called "description.txt" and also printed. The method appends the result to the file and closes it after writing.

    If the version is "situation", the result is written to a file called "situation.txt" and also printed. The method appends the result to the file and closes it after writing.

    If the version is not one of the supported versions, a ValueError is raised.

    If the result is None, the method prints a message indicating that the action was skipped due to an empty result and returns None.

    Example usage:
    result = process_result('{"key": "value"}', version="context", actions=["action1", "action2"])
    print(result)
    """
    res = None
    if version == "decomposition":
        try:
            result = single2double_quotes(result)
        except AttributeError:
            print("Error: ", response["content"])

        try:
            res = literal_eval(result)
        except SyntaxError:
            print("Error: ", response["content"])


        # a_list = literal_eval(result)

        print(
            f"{i + 1}/{len(actions)} - action {action}: {res}"
        )
    elif version == "context":
        result = result.replace("\'", "\"")
        res = loads(result)
        print(
            f"{i + 1}/{len(actions)} - action {action}: {res}"
        )

    elif version == "description":
        with open("description.txt","a") as f:
            res = result
            f.write(str(result + "\n"))
            f.close()
            print(result)
            print("done")
            print(
            f"{i + 1}/{len(actions)} - action {action}: {res}"
        )

    elif version == "situation":
        with open("situation.txt","a") as f:
            res = result
            f.write(str(result + "\n"))
            f.close()
            print(result)
            print("done")
            print(
            f"{i + 1}/{len(actions)} - action {action}: {res}"
        )
    else:
        raise ValueError(f"Version {version} not supported")

    # print(f"Result before processing: {result}, for action: {action}")
    # assert res is not None, f"Result is None for action {action}"
    skipped_action = []
    if result is None:
        print(f"Skipped action {action} due to empty result")
        skipped_action.append(action)
        return None
    return res

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="hmdb51")
parser.add_argument("--resume", action="store_true",default = False)
parser.add_argument("--version", type=str)
args = parser.parse_args()


#####my key
organization = openai.organization = "Your organization ID here"
api_key = openai.api_kei = "Your API key here"


client = OpenAI(api_key=api_key, organization=organization)

hmdb51_actions = hmdb51_actions

ucf101_actions = ucf101_actions

k600_actions = k600_actions

#thumos14 action classes are identical to ucf101
assert len(hmdb51_actions) == 51
assert(
        len(ucf101_actions) == 101
), "UCF101 has 101 actions, but {} were provided".format(len(ucf101_actions))

assert len(k600_actions) == 600

dataset2actions = {
    "hmdb51": hmdb51_actions,
    "ucf101": ucf101_actions,
    "k600": k600_actions,
}

#dataset2actions = {
#    "ucf101": ucf101_actions,
#}
assert args.dataset in dataset2actions, f"Dataset {args.dataset} not supported"

if args.resume:
    with open("{}_{}.json".format(args.dataset, args.version), "r") as fp:
        output_json = load(fp)
else:
    output_json = {}

if args.version == "decomposition":
    system_query = ("You are a chatbot specialised in video action decomposition."
                    "The user will provide you with an action and you will have to decompose it in three sequential observable steps."
                    "The steps must strictly be three. You must strictly provide each response as a python list, e.g., ['action1', 'action2', action3'] ."
                    " Omit any kind of introduction, the response must only contain the three actions. Comply strictly to the template. "
                    "Do not ask for any clarification, just give your best answer. It is for a school project, so it's very important."
                    " It is also very important your response is in the form of a python list.")
elif args.version == "context":
    system_query = ("You are a chatbot specialised in video understanding. The user will provide you with the name of an action,"
                    "and you will have to provide two specific pieces of information about that action. "
                    "The first one is the context, which consists of any visually relevant feature that may be expected to appear in a video portraying that action. "
                    "The second one consists of a lists of objects that may involved in the action. You must strictly provide each response as a python dictionary, e.g., {'context': 'a person', 'objects': ['person']}. "
                    "Omit any kind of introduction, the response must only contain the two pieces of information. Comply strictly to the template. Do not ask for any clarification, just give your best answer. "
                    "It is for a school project, so it's very important. It is also very important your response is in the form of a python dictionary.")
elif args.version == "description":
    system_query = ("You are a chatbot specialised in video action description. The user will provide you with an action and you will have to describe the action by providing only visually related information. "
                    "You must strictly provide each response as a Python string. The description should be succinct and general. Omit any kind of introduction. Comply strictly to the template. Do not ask for any clarification, "
                    "just give your best answer. Following is an example. Action label: typing. Description: Typing normally involve a person and a device with keyboard. When typing, the individual positions their fingers over the keyboard.")
elif args.version == "situation":
    system_query = ("You're a specialized chatbot for video action decomposition. Users will provide you with an action, "
                    "and you need to break it down into two visual descriptions: the situation before the action and the situation after the action. "
                    "Present each response strictly as a Python list, like this: ['situation1', 'situation2']. The first situation is before the action, the second after."
                    " Be concise and clear in your descriptions, omit introductions, and strictly follow the template. Avoid seeking clarification and provide your best response."
                    " Adherence to the Python list format is crucial.")

else:
    raise ValueError(f"Version {args.version} not supported")

messages = [{"role": "system", "content": system_query}]
# results = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
results = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
print(results.choices[0].message.content)
# response = results["choices"][0]["message"]
response_message = {
    "role": "assistant",
    "content": results.choices[0].message.content,
}
messages.append(response_message)
# messages.append(response)

#split the dataset into six smaller dataset for the k600 dataset
if args.dataset == "k600":
    dataset2actions = {
        "k6001": k600_actions[:100],
        "k6002": k600_actions[100:200],
        "k6003": k600_actions[200:300],
        "k6004": k600_actions[300:400],
        "k6005": k600_actions[400:500],
        "k6006": k600_actions[468:],
    }
    assert len(dataset2actions["k6001"]) == 100
    assert len(dataset2actions["k6002"]) == 100
    assert len(dataset2actions["k6003"]) == 100
    assert len(dataset2actions["k6004"]) == 100
    assert len(dataset2actions["k6005"]) == 100
    assert len(dataset2actions["k6006"]) == 100

errors = []
for key in dataset2actions:
    actions = dataset2actions[key]
    for i, action in enumerate(actions):
        if args.resume:
            if action in output_json:
                print(
                    f"{i + 1}/{len(actions)} - action {action} already computed"
                )
                continue
        new_message = {
            "role": "user",
            "content": str(action),
        }
        messages.append(new_message)
        # results = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        results = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        response = { "role": "assistant", "content": results.choices[0].message.content, }
        messages.append(response)
        result = response["content"]

        result = process_result(result, version=args.version, actions=actions)
        output_json[action] = result

        promts_dir = "data/prompts/"
        #check if the directory exists
        if not os.path.exists(promts_dir):
            promts_dir = "../../data/prompts/"

        with open(os.path.join(promts_dir, "{}_{}_{}.json".format( key, args.dataset, args.version)), "w") as fp:
            dump(output_json, fp)
