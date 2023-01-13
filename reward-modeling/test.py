import torch
import os

def split_ckpt(num_chunks)
    ckpt_path = "/fsx/alex/ckpts/gpt2-rm/hf_ckpt"
    sd = torch.load(os.path.join(ckpt_path, "hf_ckpt.pt"))
    keys = list(sd.keys())
    num_batches = (len(keys) - num_chunks + 1) // num_chunks
    key_batches = [keys[num_chunks*i : num_chunks * (i+1)] for i in range(num_batches)]
    index = {}
    for i, key_batch in enumerate(key_batches):
        sub_dict_name = "hf_ckpt_{}.pt".format(i)
        sub_dict = {key: sd[key] for key in key_batch}
        for key in key_batch:
            index[key] = sub_dict_name
        torch.save(sub_dict, os.path.join(ckpt_path, sub_dict_name))
        del sub_dict