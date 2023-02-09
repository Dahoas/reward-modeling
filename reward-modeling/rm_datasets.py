from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# Anthropic fine-tunes language model on entire dialogue, not just responses
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
            for ele in tqdm(data):
                prompt, response = ele["prompt"], ele["response"]
                prompt_encoding_len = len(tokenizer(prompt + "\n\n")["input_ids"])
                encodings_dict = tokenizer(prompt + "\n\n" + response + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length")
                input_id = torch.tensor(encodings_dict['input_ids'])
                attn_mask = torch.tensor(encodings_dict['attention_mask'])
                self.input_ids.append(input_id)
                self.attn_masks.append(attn_mask)
                self.labels.append(input_id)
                self.prompts.append(prompt)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx], self.labels[idx], self.prompts[idx]


# Only predicts on response tokens
class MaskedSFTDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []
            self.prompts = []
            EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]

            max_length = min(1024, max([len(tokenizer.encode(ele["prompt"] + "\n\n" + ele["response"] + '<|endoftext|>')) for ele in data]))
            print("Max length: {}".format(max_length))

            # Data expected in prompt response pairs
            for ele in tqdm(data):
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


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length, max_num=-1):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        PAD_ID = tokenizer.pad_token

        for i, pair in enumerate(tqdm(pairs)):
            if max_num >= 0 and i > max_num:
                break
            prompt = pair["prompt"]
            chosen, rejected = pair["chosen"], pair["rejected"]
            tok_chosen = tokenizer(prompt + chosen + "<|endoftext|>", return_tensors="pt")["input_ids"]
            tok_rejected = tokenizer(prompt + rejected + "<|endoftext|>", return_tensors="pt")["input_ids"]
            # Reject data with num tokens > max_length
            if tok_chosen.shape[-1] <= max_length and tok_rejected.shape[-1] <= max_length and chosen != rejected:
                chosen_encodings_dict = tokenizer(prompt + chosen + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length", return_tensors="pt")
                rejected_encodings_dict = tokenizer(prompt + rejected + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length", return_tensors="pt")
                self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
                self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
                self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
                self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], self.rejected_attn_masks[idx]


class PairwiseEvalDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []

        for pair in tqdm(pairs):
            prompt = pair["prompt"]
            chosen, rejected = pair["chosen"], pair["rejected"]
            tok_chosen = tokenizer(prompt + chosen + "<|endoftext|>", return_tensors="pt")["input_ids"]
            tok_rejected = tokenizer(prompt + rejected + "<|endoftext|>", return_tensors="pt")["input_ids"]
            # Reject data with num tokens > max_length
            if tok_chosen.shape[-1] <= max_length and tok_rejected.shape[-1] <= max_length:
                chosen_encodings_dict = tokenizer(prompt + chosen + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length", return_tensors="pt")
                rejected_encodings_dict = tokenizer(prompt + rejected + '<|endoftext|>', truncation=True,
                                        max_length=max_length, padding="max_length", return_tensors="pt")
                # First append chosen then rejected
                self.input_ids.append(chosen_encodings_dict['input_ids'])
                self.attn_masks.append(chosen_encodings_dict['attention_mask'])
                self.input_ids.append(rejected_encodings_dict['input_ids'])
                self.attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


class RankedDataset(Dataset):
    def __init__(self, rankings, tokenizer, max_length, order=["instruct", "20B", "6B", "1B", "125M"], max_num=-1):
        self.rankings_input_ids = []
        self.rankings_attn_masks = []
        PAD_ID = tokenizer.pad_token

        for i, ranks in enumerate(tqdm(rankings)):
            if max_num >= 0 and i > max_num:
                break
            prompt = ranks.pop("prompt")
            toks = []
            long = False
            for response in ranks.values():
                long = long or len(tokenizer(prompt + response + "<|endoftext|>", return_tensors="pt")["input_ids"]) > max_length
            if long:
                continue
            # Reject data with num tokens > max_length
            ranking_input_ids = []
            ranking_attn_masks = []
            for model in order:
                response = ranks[model]
                encodings_dict = tokenizer(prompt + response + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
                ranking_input_ids.append(encodings_dict['input_ids'].view(1, -1))
                ranking_attn_masks.append(encodings_dict['attention_mask'].view(1, -1))
            ranking_input_ids = torch.cat(ranking_input_ids, dim=0)
            ranking_attn_masks = torch.cat(ranking_attn_masks, dim=0)
            self.rankings_input_ids.append(ranking_input_ids)
            self.rankings_attn_masks.append(ranking_attn_masks)

    def __len__(self):
        return len(self.rankings_input_ids)

    def __getitem__(self, idx):
        return self.rankings_input_ids[idx], self.rankings_attn_masks[idx]


class RankedEvalDataset(Dataset):
    def __init__(self, rankings, tokenizer, max_length, order=["instruct", "20B", "6B", "1B", "125M"], max_num=-1):
        self.rankings_input_ids = []
        self.rankings_attn_masks = []
        PAD_ID = tokenizer.pad_token

        for i, ranks in enumerate(tqdm(rankings)):
            if max_num >= 0 and i > max_num:
                break
            prompt = ranks.pop("prompt")
            toks = []
            long = False
            for response in ranks.values():
                long = long or len(tokenizer(prompt + response + "<|endoftext|>", return_tensors="pt")["input_ids"]) > max_length
            if long:
                continue
            # Reject data with num tokens > max_length
            ranking_input_ids = []
            ranking_attn_masks = []
            for model in order:
                response = ranks[model]
                encodings_dict = tokenizer(prompt + response + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
                ranking_input_ids.append(encodings_dict['input_ids'].view(1, -1))
                ranking_attn_masks.append(encodings_dict['attention_mask'].view(1, -1))
            self.rankings_input_ids += ranking_input_ids
            self.rankings_attn_masks += ranking_attn_masks

    def __len__(self):
        return len(self.rankings_input_ids)

    def __getitem__(self, idx):
        return self.rankings_input_ids[idx], self.rankings_attn_masks[idx], None


def pairwise_data_collator(data):
    if len(data[0]) == 4:
        return {'input_ids': torch.cat([f[0] for f in data] + [f[2] for f in data]),
                'attention_mask': torch.cat([f[1] for f in data] + [f[3] for f in data])}
    elif len(data[0]) == 2:
        return {'input_ids': torch.cat([f[0] for f in data]),
                'attention_mask': torch.cat([f[1] for f in data])}
    else:
        raise ValueError("Invalid data format")


def ranked_data_collator(data):
    return {
            "input_ids": torch.cat([f[0] for f in data], dim=0),
            "attention_mask": torch.cat([f[1] for f in data], dim=0),
        }
