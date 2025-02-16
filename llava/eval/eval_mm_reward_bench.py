import argparse
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
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
# import debugpy


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def process_video(video_file, image_processor, config):
    device = 'cuda'
    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
    frame_files.sort()
    sample_frames = 32
    sample_indices = np.linspace(0, len(frame_files) - 1, sample_frames, dtype=int)

    # 确保所有索引都在合法范围内（使用 np.clip）
    sample_indices = np.clip(sample_indices, 0, len(frame_files) - 1)

    # 获取采样的 frame_files
    frame_files = [frame_files[i] for i in sample_indices]

    # 打开并转换为 RGB 图像
    frame_files = [Image.open(item).convert("RGB") for item in frame_files]
    
    config.image_aspect_ratio = 'pad'
    image_tensor = process_images(frame_files, image_processor, config)
    
    return image_tensor

def make_conv_rm(prompt, answer, has_image=True):
    critic_prompt = (
        "You are an unbiased and fair evaluator tasked with assessing the quality of answers provided by a Large Multimodal Model (LMM). "
        "Given the following question and the LMM's response, please evaluate the correctness, clarity, and completeness of the answer. "
        "Provide a detailed explanation of your assessment, including specific points of strength and areas for improvement.\n\n"
        "[Question]: {question}\n"
        "[LMM Response]: {answer}\n\n"
        "[Evaluation]:"
    )
    formatted_prompt = critic_prompt.format(question=prompt, answer=answer)
    if has_image:
        formatted_prompt = formatted_prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        formatted_prompt = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt
        formatted_prompt = formatted_prompt.strip()
    return formatted_prompt

def eval_model(args):
    # Model
    disable_torch_init()
    # model_name = "llava"
    device_map = "auto"
    model_path = os.path.expanduser(args.model_path)
    #yt
    model_name = "llava_qwen"
    device = "cuda"

    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map, attn_implementation=None)
    model.eval()

    questions = []
    with open(args.question_file, 'r') as file:
        for line in file:
            questions.append(json.loads(line))
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
    ans_file = open(answers_file, "a+")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    index, cnt_images = 0, []
    for line in tqdm(questions, total=len(questions)):
        video_file = line.get("video", None)
        image_file = line.get("image", None)
        # image_file = line["image"]
        
        if image_file:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]
            size = [image.size]
        else:
            image_tensor = process_video(video_file, image_processor, model.config)
            size = None

        if video_file:
            modalities=["video"]
        if image_file:
            modalities=["image"]

        if isinstance(image_tensor, tuple):
            image_tensor = image_tensor[0]
        if isinstance(image_tensor, list):
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=torch.float16, device=device)
        answers = [line['chosen'], line['rejected']]
        critics = [line['chosen_reason'], line['rejected_reason']]
        rewards = []
        critic_texts = []
        for t in range(len(answers)):
            qs = make_conv_rm(line["prompt"], answers[t], video_file or image_file)
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            
            critic_reward = []
            
            for critic_t in range(args.num_critic):
                if not args.use_gt_critic and not args.wo_critic:
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            image_sizes=size,
                            do_sample=True if args.temperature > 0 else False,
                            modalities=modalities,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True)
                        
                    critic = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    critic_texts.append(critic)
                elif not args.wo_critic:
                    critic = critics[t]
                else:
                    critic = None
                
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], critic)
                prompt = conv.get_prompt()

                input_ids_reward = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
                with torch.inference_mode():
                    reward, _ = model(
                        input_ids_reward,
                        images=image_tensor,
                        image_sizes=size,
                        rm_forward=True)
                    critic_reward.append(reward[0].item())
            
            rewards.append(np.mean(critic_reward))
        
        # import pdb;pdb.set_trace()
        index += 1
        line['rewards'] = rewards
        line['critic'] = critic_texts
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling strategy follows simpo
    # nohup python scripts/reward_model/ours_cirtic_eval.py --num-chunks 4 --chunk-idx 0 >> /mllm_hdd/mllm_hdd/yfzhang/data/alignment/vl_reward/0_4.log 2>&1 &
    # parser.add_argument("--rm-model-path", type=str, default="../model/alignment/llava16_llava_rlhf_reward_model_lr1e_5_bsz128_freevision_reward_model_coefficent")
    parser.add_argument("--model-path", type=str, default="yifanzhang114/MM-RLHF-Reward-7B-llava-ov-qwen")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mllm_hdd/mllm_hdd/yfzhang/data/alignment/")
    parser.add_argument("--question-file", type=str, default="./mm_reward_bench.jsonl")
    parser.add_argument("--answers-file", type=str, default="./mm_reward_bench_result.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_critic", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--use_gt_critic", type=bool, default=False)
    parser.add_argument("--wo_critic", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")

    parser.add_argument(
        "--test-prompt",
        type=str,
        default="",
    )
    args = parser.parse_args()
    print(args)

    eval_model(args)