from torch.utils.data import Dataset

class TextDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []
            EOS_ID = tokenizer("<|endoftext|>")

            max_length = max([len(tokenizer.encode(ele["prompt"] + "\n\n" + ele["response"] + '<|endoftext|>')) for ele in data])
            print("Max length: {}".format(max_length))

            # Data expected in prompt response pairs
            for ele in data:
                prompt, response = ele["prompt"], ele["response"]
                prompt_encoding_len = len(tokenizer(prompt + "\n\n")["input_ids"])
                encodings_dict = tokenizer(prompt + "\n\n" + response + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                label_mask = self.input_ids[-1] == EOS_ID
                first_eos = eos_mask.nonzero()[0, 0]
                label_mask[first_eos] = 0  # Want to predict on first eos_token
                label_mask[:prompt_encoding_len] = 1  # Do not predict on prompt
                flipped_mask = 1 - label_mask
                self.labels.append(self.input_ids * flipped_mask - 100 * label_mask)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]