import torch
import torch.nn.functional as F
import math
import statistics
import time


def move_tokens_to_piles(page_list, pile, tensor, pages_per_seq, page_size, seq_len, batch_size, num_kv_heads, head_dim):
    pile_padded = F.pad(tensor, (0, 0, 0, pages_per_seq * page_size - seq_len), "constant", 0).reshape(batch_size, num_kv_heads, pages_per_seq, page_size, head_dim)
    pile_padded = pile_padded.permute(0, 2, 1, 3, 4)
    pile_padded = torch.flatten(pile_padded, start_dim=0, end_dim=1)
    pile[page_list, :, :, :] = pile_padded
    return pile

def create_data(head_dim, num_kv_heads, num_query_heads, batch_size, seq_len, page_size, device):
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    query = torch.randn(batch_size, num_query_heads, 1, head_dim, dtype=torch.float16, device=device)

    pages_per_seq = math.ceil(seq_len / page_size)
    total_pages = batch_size * pages_per_seq

    key_pile = torch.zeros((total_pages, num_kv_heads, page_size, head_dim), dtype=torch.float16, device=device)
    value_pile = torch.zeros((total_pages, num_kv_heads, page_size, head_dim), dtype=torch.float16, device=device)

    page_list = torch.randperm(total_pages, dtype=torch.int32, device=device)
    
    key_pile = move_tokens_to_piles(page_list, key_pile, key, pages_per_seq, page_size, seq_len, batch_size, num_kv_heads, head_dim)
    value_pile = move_tokens_to_piles(page_list, value_pile, value, pages_per_seq, page_size, seq_len, batch_size, num_kv_heads, head_dim)

    page_list = page_list.reshape(batch_size, pages_per_seq)
    seq_len_tensor = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    return {
        "key": key,
        "value": value,
        "query": query,
        "key_pile": key_pile,
        "value_pile": value_pile,
        "page_list": page_list,
        "seq_len_tensor": seq_len_tensor
    }

def check_piles(data):
    for name in ("key", "value"):
        original = data[name]
        pile = data[name + "_pile"]
        batch, heads, seq_len, head_dim = original.shape

        pages = pile[data["page_list"]] # [batch, pages_per_seq, heads, page_size, head_dim]
        rebuilt = pages.permute(0, 2, 1, 3, 4).reshape(batch, heads, -1, head_dim)
        rebuilt = rebuilt[:, :, :seq_len, :]# drop the padding

        if not torch.equal(rebuilt, original):
            print(f"FAIL: {name} pile doesn't match")
            return False

    print("PASS")
    return True

def get_answer(key, value, query):
    batch_size = key.shape[0]
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_dim = key.shape[-1]

    attention_scores = []
    for batch in range(batch_size):
        seq_key= key[batch, :, :, :].to(torch.float32)
        seq_value = value[batch, :, :, :].to(torch.float32)
        seq_query = query[batch, :, :, :].to(torch.float32)
    
        repeat = int(num_query_heads / num_kv_heads)
        seq_key = torch.repeat_interleave(seq_key, repeats=repeat, dim=0)
        seq_value = torch.repeat_interleave(seq_value, repeats=repeat, dim=0)

        attention = torch.softmax((seq_query @ seq_key.transpose(-2, -1)) * (1 / math.sqrt(head_dim)), dim=-1) @ seq_value
        attention_scores.append(attention)

    attention_scores = torch.stack(attention_scores, dim=0)
    return attention_scores

def check_answer(key, value, query, answer):
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    query = query.to(torch.float32)
    sdpa_answer = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)

    if not torch.allclose(answer, sdpa_answer, atol=1e-4):
        print(f"FAIL: answers do not match")
        return False
    
    print("PASS")
    return True

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_kernel(fn):
    for _ in range(10):
        fn()
    
    if torch.cuda.is_available():
        throw_away_tensor = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    total = []
    for _ in range(100):
        if torch.cuda.is_available():
            throw_away_tensor.zero_()
        sync()
        start = time.perf_counter()
        fn()
        sync()
        end = time.perf_counter()
        total.append((end - start) * 1000)

    return statistics.median(total)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = create_data(head_dim=128, num_kv_heads=8, num_query_heads=32, batch_size=64, seq_len=4096, page_size=16, device=device)
    check_piles(data)
    attention_scores = get_answer(data["key"], data["value"], data["query"])
    check_answer(data["key"], data["value"], data["query"], attention_scores)
    ms = time_kernel(lambda: get_answer(data["key"], data["value"], data["query"]))
    print(f"{ms} ms")
