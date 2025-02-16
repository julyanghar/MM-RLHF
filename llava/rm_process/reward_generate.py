import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, prompt):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt = prompt

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        questions = line["outputs"]
        input_ides = []
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # self.model_config.image_aspect_ratio = 'pad_and_divide'
        # image_tensor = process_images([image], self.image_processor, self.model_config)
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        for qs in questions:
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                    # print(qs)
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ides.append(input_ids)
        # input_ides = torch.stack(input_ides)
        return input_ides, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids[0], image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, prompt=''):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    rm_model_path = os.path.expanduser(args.model_path)
    device_map = "auto"
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map)
    model.eval()

    file_paths = [
        "tmp/llava_16_7b_iter1_chunk0_of_6.jsonl",
        "tmp/llava_16_7b_iter1_chunk1_of_6.jsonl",
        "tmp/llava_16_7b_iter1_chunk2_of_6.jsonl",
        "tmp/llava_16_7b_iter1_chunk3_of_6.jsonl",
        "tmp/llava_16_7b_iter1_chunk4_of_6.jsonl",
        "tmp/llava_16_7b_iter1_chunk5_of_6.jsonl"
    ]

    # 初始化一个空列表来存储每个文件的内容
    questions = []

    # 逐个读取文件并添加内容到列表中
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = [json.loads(line) for line in file]  # 逐行读取并解析为JSON对象
            questions.extend(file_content)
            
    # questions = []
    # with open(args.question_file, 'r') as file:
    #     for line in file:
    #         questions.append(json.loads(line))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file_with_chunks = f"{args.answers_file.rsplit('.', 1)[0]}_chunk{args.chunk_idx}_of_{args.num_chunks}.jsonl"
    existing_ids = set()
    try:
        with open(answers_file_with_chunks, 'r') as file:
            for line in file:
                answer = json.loads(line)
                existing_ids.add(answer.get("id"))
    except FileNotFoundError:
        print(f"No existing file found: {answers_file_with_chunks}, proceeding with empty set of existing IDs.")

    questions = [q for q in questions if q.get("id") not in existing_ids]
    print(f'saving all the answers to {answers_file_with_chunks}')
    answers_file = os.path.expanduser(answers_file_with_chunks)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, prompt=args.test_prompt)

    index, cnt_images = 0, []
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["id"]
        cur_prompt = line["outputs"]

        cnt_images.append(image_tensor.shape[0])
        output_texts = []
        for t in range(5):
            input_id = input_ids[t].unsqueeze(0)
            input_id = input_id.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                reward = model(
                    input_id,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    rm_forward=True)
                
            output_texts.append(reward[0].cpu().item())
        
        index += 1
        ans_id = shortuuid.uuid()
        line['rewards'] = output_texts
        output_texts = np.array(output_texts)
        idx_max, idx_min = output_texts.argmax(), output_texts.argmin()
        line['chosen'] = cur_prompt[idx_max]
        line['rejected'] = cur_prompt[idx_min]
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling strategy follows simpo
    # nohup python tmp/reward_generate.py >> tmp/llava16_reward_llavaov_generate_and_select.log 2>&1 &
    parser.add_argument("--model-path", type=str, default="../model/alignment/llavaov_qwen_llava_rlhf_reward_model_lr1e_5_bsz128_freevision_reward_model_coefficent")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/llava_sft/coco/train2017")
    parser.add_argument("--question-file", type=str, default="tmp/llava_16_7b_iter1_chunk0_of_6.jsonl")
    parser.add_argument("--answers-file", type=str, default="./tmp/llava_16_7b_rewardllavaov_iter1_dpo.jsonl")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")

    parser.add_argument(
        "--test-prompt",
        type=str,
        default="",
    )
    args = parser.parse_args()
    print(args)

    eval_model(args)
    