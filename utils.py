import tiktoken


def validate_token_length(texts, model_name: str = "text-embedding-3-small"):
    encoder = tiktoken.encoding_for_model(model_name)
    valid = []
    for text in texts:
        tokens = encoder.encode(text)
        if 1 <= len(tokens) <= 8191:  # OpenAI 的限制
            valid.append(text)
    return valid


# 将文本分批处理（关键优化）


def batch_embed(texts, model_name="text-embedding-3-small", max_tokens=8192 * 8):
    encoder = encoding_for_model(model_name)
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        # 截断处理逻辑
        tokens = truncate_to_tokens(model_name, text, 8191)

        # 剩余逻辑保持不变
        if current_tokens + len(tokens) > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += len(tokens)

    if current_batch:
        batches.append(current_batch)

    return batches


from tiktoken import encoding_for_model


def validate_batch(texts, model_name):
    problematic = []
    for i, text in enumerate(texts):
        try:
            encoder = encoding_for_model(model_name)  # 根据你的模型选择
            tokens = encoder.encode(text)
            if len(tokens) > 8191:
                problematic.append((i, "token超限"))
            elif len(tokens) == 0:
                problematic.append((i, "空文本"))
        except Exception as e:
            problematic.append((i, f"编码错误: {str(e)}"))
    return problematic


# 执行验证
def my_test(valid_texts, model_name):
    issues = validate_batch(valid_texts, model_name)
    if issues:
        print(f"发现 {len(issues)} 处问题，首5条示例:")
        for idx, reason in issues[:5]:
            print(f"索引 {idx}: {reason}")
            print(f"问题文本预览: {repr(valid_texts[idx][:50])}")


def truncate_to_tokens(model_name, text, max_tokens):
    encoder = encoding_for_model(model_name)
    tokens = encoder.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return encoder.decode(truncated_tokens)
