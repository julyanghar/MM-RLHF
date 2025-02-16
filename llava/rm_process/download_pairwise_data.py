from datasets import load_dataset
import json
input_file  = '/data/Alignment/llava_7b_v1_preference.json'
output_file = 'tmp/llava_7b_v1_preference.jsonl'

# 打开并读取文件
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    datas = json.load(f_in)
    for data in datas:
        
        human_prompt = next(item['value'] for item in reversed(data['conversations']) if item['from'] == 'human')

        # Get the chosen and rejected outputs based on preference
        chosen = data['output_1'] if data['preference'] == 1 else data['output_2']
        rejected = data['output_2'] if data['preference'] == 1 else data['output_1']

        # Create the modified data structure
        modified_data = {
            "prompt": human_prompt,
            "chosen": chosen["value"],
            "rejected": rejected["value"],
            "has_image": True,
            "image": data['image'],
            "id": data['id']
        }

        # Write the modified data to the output file
        f_out.write(json.dumps(modified_data) + '\n')