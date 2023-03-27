import argparse, time
import openai
import json
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unexpected_contents')
    parser.add_argument('--openai_api_key', default=None, type=str, required=True, help="API key to use GPT-3.")
    # parser.add_argument('--preprocess', action='store_true', help="preprocess the ToMi dataset")
    parser.add_argument('--predict', action='store_true', help="Requests openai to generate responses")
    args = parser.parse_args()
    opt = vars(args)
    openai.api_key = args.openai_api_key
    input_path = f"additional_data/{opt['dataset']}.jsonl"
    continue_index = 0
    if opt['predict']:
        predict(input_path, continue_index, opt)

    accuracy(opt)
    # gold, predictions = average_accuracy(opt)
    # joint_accuracy(gold, predictions)


def accuracy(opt):
    gold = []
    predictions = []
    if opt['dataset'] == 'unexpected_transfer':
        with open(f'output/{opt["dataset"]}.txt') as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                fields = json.loads(line)
                gold.append([fields[fields['truth']], fields[fields['belief']], fields[fields['belief']]])

                predictions.append([fields['prediction1'], fields['prediction2'], fields['prediction3']])
        gold = np.array(gold)
        print(gold)
        predictions = np.array(predictions)
        accuracy_q1 = accuracy_score(gold[:, 0], predictions[:, 0])
        accuracy_q2 = accuracy_score(gold[:, 1], predictions[:, 1])
        accuracy_q3 = accuracy_score(gold[:, 2], predictions[:, 2])
        total_accuracy = (accuracy_q1 + accuracy_q2 + accuracy_q3) / 3
        joint_accuracy = 0
        for i in range(gold.shape[0]):
            joint_accuracy += int(accuracy_score(gold[i], predictions[i]))
        joint_accuracy /= gold.shape[0]

        print(f"q1 accuracy: {accuracy_q1: .3f}")
        print(f"q2 accuracy: {accuracy_q2: .3f}")
        print(f"q3 accuracy: {accuracy_q3: .3f}")
        print(f"Average accuracy: {total_accuracy: .3f}")
        print(f"Joint `accuracy: {joint_accuracy: .3f}")
    elif opt['dataset'] == 'unexpected_contents':
        with open(f'output/{opt["dataset"]}.txt') as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                fields = json.loads(line)
                print(fields[fields['belief']] + "     " + fields['prediction1'])
                gold.append([fields[fields['belief']]])
                predictions.append([fields['prediction1']])
        gold = np.array(gold)
        predictions = np.array(predictions)
        accuracy_q1 = accuracy_score(gold[:, 0], predictions[:, 0])
        print(f"q1 accuracy: {accuracy_q1: .3f}")


def predict(input_path, continue_index, opt):
    preprompt = get_preprompt(opt)
    with open(f"output/{opt['dataset']}.txt", "a") as f_out:
        with open(input_path) as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                if i < continue_index:
                    continue
                print(i)
                fields = json.loads(line)
                prompts = get_prompt(preprompt, opt, fields)
                for i, prompt in enumerate(prompts):
                    fields['prediction' + str(i+1)] = open_ai_finalanswer_request(prompt, i, 0).strip()
                fields['prompts'] = prompts
                f_out.write(json.dumps(fields) + "\n")

def get_preprompt(opt):
    prompt = ""
    if opt['dataset'] == 'unexpected_contents' or opt['dataset'] == 'unexpected_transfer':
        with open(f"additional_data/{opt['dataset']}_preprompt.txt") as f_in:
            for line in f_in:
                data = json.loads(line)
                prompt += 'Context: ' + data['context'] + '\n' + 'Question: Fill in the blanks with the best option. ' \
                          + data['question'] + '\n' + f"- {data['o1']}\n" + f"- {data['o2']}\n" + \
                          f"Answer: {data['label']}" + "\n\n"
    return prompt


def get_prompt(preprompt, opt, fields):
    prompts = []
    if opt['dataset'] == 'unexpected_transfer':
        context = fields['txt'].strip()
        choices = '- ' + fields['o1'].strip() + '\n' + '- ' + fields['o2'].strip()
        for i in range(1, 4):
            question = 'Question: Fill in the blank with the best option. ' + fields['q' + str(i)].strip() + " _"
            prompt = preprompt + "Context: " + context + '\n' + question + '\n' + choices + '\n' + "Answer:"
            prompts.append(prompt)
    if opt['dataset'] == 'unexpected_contents':
        # preprompt += f"[o1] = {fields['o1']}\n[o2] = {fields['o2']}\n[ctr] = {fields['ctr']}\n"
        context = fields['txt'].strip().replace("[o1]", fields['o1'])
        context = context.replace("[o2]", fields['o2'])
        context = context.replace("[ctr]", fields['ctr'])
        choices = '- ' + fields['o1'].strip() + '\n' + '- ' + fields['o2'].strip()
        question = 'Question: Fill in the blank with the best option. [he/she] ' + fields['q3'].strip() + " _"
        prompt = preprompt + "Context: " + context + '\n' + question + '\n' + choices + '\n' + "Answer:"
        prompts.append(prompt)
    return prompts


def open_ai_finalanswer_request(prompt, i, counter):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=30,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text'].strip()
    except:
        if counter < 3:
            time.sleep(10)
            return open_ai_finalanswer_request(prompt, i, counter + 1)
        else:
            print(prompt)
            print("continue from:" + str(i))
            exit()


if __name__ == '__main__':
    main()