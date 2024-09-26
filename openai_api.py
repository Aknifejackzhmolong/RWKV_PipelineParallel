################################################################
# 仅供测试，采用Flask框架，使用前先修改模型参数。
# 支持少量并发（居然能支持是我没想到的，应该是Flask框架自己支持）
# api_url填写http://127.0.0.1:8848即可测试
# 参数基本符合OpenAI的接口，用任意OpenAI客户端均可，无需填写api key和model参数
###############################################################
from flask import Flask, request, Response, jsonify
from flask import stream_with_context
from src.model import RWKV_RNN,ModelArgs
from src.model_utils import device_checker, device_specific_empty_cache
from src.sampler import sample_logits, apply_penalties
from src.rwkv_tokenizer import RWKV_TOKENIZER
import torch
import time
import uuid
import json
import copy

app = Flask(__name__)

# 添加跨域头信息的装饰器
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# 初始化模型和分词器
def init_model():
    # 模型参数配置
    with open("train/params-small.json", "r") as f:
        args = ModelArgs.from_dict(json.load(f))
        args = device_checker(args)
        device = args.device
        assert device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']
        print(f"Device: {device}")


    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    # 初始化状态
    global_state = model.init_state(1)

    tokenizer = RWKV_TOKENIZER(args.TOKENIZER_PATH)
    print("Done")
    print(f"Model name: {args.MODEL_NAME.split('/')[-1]}")
    return model, tokenizer, global_state, device, args

def format_messages_to_prompt(messages):
    formatted_prompt = ""

    # Define the roles mapping to the desired names
    role_names = {
        "system": "System",
        "assistant": "Assistant",
        "user": "User"
    }

    # Iterate through the messages and format them
    for message in messages:
        role = role_names.get(message['role'], 'Unknown')  # Get the role name, default to 'Unknown'
        content = message['content']
        formatted_prompt += f"{role}: {content}\n\n"  # Add the role and content to the prompt with newlines

    formatted_prompt += "Assistant: "
    return formatted_prompt

# 生成文本的函数
def generate_text(prompt: str, temperature=1.5, top_p=0.1, max_tokens=2048, presence_penalty=0.0,
                  frequency_penalty=0.0, stop=['\n\nUser', '<|endoftext|>']):
    """
    使用模型生成文本。

    Args:
        prompt (str): 初始字符串。
        temperature (float): 控制生成文本多样性的温度参数。默认值为 1.5。
        top_p (float): 控制采样概率分布的 top-p 截断参数。默认值为 0.1。
        max_tokens (int): 控制生成文本的最大标记数。默认值为 2048。
        stop (list of str): 停止条件列表。默认为 ['\n\nUser']。
    Returns:
        generated_tokens (str): 生成的文本字符串。
        if_max_token (bool): 是否达到了最大标记数的布尔值。
        usage (dict): token使用量计算。
    """
    # 设置续写的初始字符串和参数
    encoded_input = tokenizer.encode([prompt])
    token = torch.tensor(encoded_input).long().to(device)
    state = copy.deepcopy(global_state)
    state = state.to(device)
    prompt_tokens = len(encoded_input[0])
    stop_token = tokenizer.encode(stop)[0]

    if args.parallel:
        with torch.no_grad():
            token_out, state = model.forward_parallel_slices(token, state, slice_len=512)
            out = token_out[:, -1]
    else:
        # 预填充状态
        token = token.transpose(0, 1).to(device)
        with torch.no_grad():
            for t in token:
                out, state = model.forward(t, state)

    del token


    completion_tokens = 0
    if_max_token = True
    generated_texts = ''
    generated_tokens = None
    freq_dict = None
    for step in range(max_tokens):
        token_sampled, generated_tokens, freq_dict = apply_penalties(
            logits=out,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            token=generated_tokens,
            freq_dict=freq_dict
        )
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)

        # 判断是否达到停止条件
        last_text = tokenizer.decode(token_sampled.unsqueeze(1).cpu().tolist())[0]
        generated_texts += last_text
        completion_tokens += 1
        print(last_text, end='')


        # 如果末尾含有 stop 列表中的字符串，则停止生成
        if generated_texts.endswith(tuple(stop)):
            for stop_word in stop:
                generated_texts = generated_texts.replace(stop_word, "") # 替换掉终止token
            if_max_token = False
            break

    total_tokens = prompt_tokens + completion_tokens
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
    del state
    clear_cache()
    return generated_texts, if_max_token, usage

# 生成文本的生成器函数
def generate_text_stream(prompt: str, temperature=1.5, top_p=0.1, max_tokens=2048, presence_penalty = 0.0,
    frequency_penalty = 0.0, stop=['\n\nUser', '<|endoftext|>']):
    encoded_input = tokenizer.encode([prompt])
    token = torch.tensor(encoded_input).long().to(device)
    state = copy.deepcopy(global_state)
    state = state.to(device)
    prompt_tokens = len(encoded_input[0])

    try:
        if args.parallel:
            with torch.no_grad():
                token_out, state = model.forward_parallel_slices(token, state, slice_len=512)
                out = token_out[:, -1]  # 取最后一个生成的token
        else:
            # 预填充状态
            token_temp = token.transpose(0, 1).to(device)
            with torch.no_grad():
                for t in token_temp:
                    out, state = model.forward(t, state)
            del token_temp  # 释放内存
        # del token
    except GeneratorExit:
        # 客户端断开连接，停止生成并清理资源
        clear_cache()
        return

    generated_texts = ''
    generated_tokens = token[0]
    completion_tokens = 0
    if_max_token = True
    freq_dict = None
    for step in range(max_tokens):
        token_sampled, generated_tokens, freq_dict = apply_penalties(
            logits=out,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            token=generated_tokens,
            freq_dict=freq_dict
        )
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)

        last_text = tokenizer.decode(token_sampled.unsqueeze(1).cpu().tolist())[0]
        generated_texts += last_text
        completion_tokens += 1

        if generated_texts.endswith(tuple(stop)):
            if_max_token = False
            response = {
                "object": "chat.completion.chunk",
                "model": "rwkv",
                "choices": [{
                    "delta": "",
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(response)}\n\n"
            clear_cache()
            break
        else:
            response = {
                "object": "chat.completion.chunk",
                "model": "rwkv",
                "choices": [{
                    "delta": {"content": last_text},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            try:
                yield f"data: {json.dumps(response)}\n\n"
            except GeneratorExit:
                # 客户端断开连接，停止生成并清理资源
                clear_cache()
                return

    if if_max_token:
        response = {
            "object": "chat.completion.chunk",
            "model": "rwkv",
            "choices": [{
                "delta": "",
                "index": 0,
                "finish_reason": "length"
            }]
        }
        yield f"data: {json.dumps(response)}\n\n"

    del state
    clear_cache()
    yield "data: [DONE]"


def clear_cache():
    device_specific_empty_cache(args)

# 处理 OPTIONS 请求
@app.route('/v1/models', methods=['GET'])
def model_request():
    return jsonify(dict(
        object='list',
        data=[
            dict(id='rwkv',object='model')
        ]
    ))
# 处理 OPTIONS 请求
@app.route('/v1/chat/completions', methods=['OPTIONS'])
def options_request():
    return Response(status=200)

# 定义流式输出的路由
# Define your completion route
@app.route('/v1/chat/completions', methods=['POST'])
def create_completion():
    try:
        # Extract parameters from the request
        data = request.json
        model = data.get('model', 'rwkv')
        messages = data['messages']
        print(data)
        stream = data.get('stream', True)
        temperature = data.get('temperature', 0.8)
        top_p = data.get('top_p', 0.3)
        presence_penalty = data.get('presence_penalty', 0.5)
        frequency_penalty = data.get('frequency_penalty', 0.5)
        max_tokens = data.get('max_tokens', 2048)
        stop = data.get('stop', ['\n\nUser', '<|endoftext|>'])

        prompt = format_messages_to_prompt(messages)

        # Determine if streaming is enabled
        if stream:
            """
            def generate():
                for event in generate_text_stream(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop):
                    yield event
            return Response(generate(), content_type='text/event-stream')
            """
            response = Response(stream_with_context(generate_text_stream(prompt, temperature=temperature, top_p=top_p, presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty, max_tokens=max_tokens, stop=stop)),
                                content_type='text/event-stream')
            response.timeout = None  # 设置超时时间为无限制
            return response
        else:
            completion, if_max_token, usage = generate_text(prompt, temperature=temperature, top_p=top_p, presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty, max_tokens=max_tokens, stop=stop)
            finish_reason = "stop" if if_max_token else "length"
            unique_id = str(uuid.uuid4())
            current_timestamp = int(time.time())
            response = {
                "id": unique_id,
                "object": "chat.completion",
                "created": current_timestamp,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion,
                    },
                    "finish_reason": finish_reason
                }],
                "usage": usage
            }
            clear_cache()
            return jsonify(response)
    except Exception as e:
        clear_cache()
        return str(e), 500

if __name__ == '__main__':
    model, tokenizer, global_state, device, args = init_model()
    app.run(host='0.0.0.0', port=8848)

