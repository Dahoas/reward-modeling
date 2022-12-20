from torch.utils.data import Dataset
import torch

class SFTDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []
            self.prompts = []
            EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]

            max_length = min(1024, max([len(tokenizer.encode(ele["prompt"] + "\n\n" + ele["response"] + '<|endoftext|>')) for ele in data]))
            print("Max length: {}".format(max_length))

            # Data expected in prompt response pairs
            for ele in data:
                prompt, response = ele["prompt"], ele["response"]
                prompt_encoding_len = len(tokenizer(prompt + "\n\n")["input_ids"])
                encodings_dict = tokenizer(prompt + "\n\n" + response + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length")
                input_id = torch.tensor(encodings_dict['input_ids'])
                attn_mask = torch.tensor(encodings_dict['attention_mask'])
                label_mask = (input_id == EOS_ID).type(torch.int32)
                first_eos = label_mask.nonzero()
                # Skip text which has no eos token
                if len(first_eos) == 0:
                    continue
                else:
                    first_eos = first_eos[0, 0]
                label_mask[first_eos] = 0  # Want to predict on first eos_token
                label_mask[:prompt_encoding_len] = 1  # Do not predict on prompt
                flipped_mask = 1 - label_mask
                self.input_ids.append(input_id)
                self.attn_masks.append(attn_mask)
                self.labels.append(self.input_ids[-1] * flipped_mask - 100 * label_mask)
                self.prompts.append(prompt)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx], self.labels[idx], self.prompts[idx]