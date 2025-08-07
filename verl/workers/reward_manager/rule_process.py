from typing import List, Tuple
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


def create_char_to_token_map(tokenizer, token_ids: torch.Tensor) -> Tuple[str, List[int]]:
    pieces = []
    for tid in token_ids.tolist():
        piece = tokenizer.decode([tid],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        pieces.append(piece)

    response_str = "".join(pieces)

    # Build char→token index map aligned to ORIGINAL token indices
    char_to_token = []
    for tok_idx, piece in enumerate(pieces):
        char_to_token.extend([tok_idx] * len(piece))

    return response_str, char_to_token


class RuleProcessRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the RuleProcessRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # added:
            special_id_set = set(self.tokenizer.all_special_ids)
            # keep an alignment list so we can write rewards back to the *original*
            # positions later
            id_map  = []  # orig-pos  -> new-pos (or -1 if special)
            clean_ids = []
            for t in valid_response_ids.tolist():
                if t in special_id_set:
                    id_map.append(-1)
                else:
                    id_map.append(len(clean_ids))
                    clean_ids.append(t)

            clean_ids = torch.tensor(clean_ids, dtype=valid_response_ids.dtype, device=valid_response_ids.device)
            
            # decode **without** skipping specials because we already removed them
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # response_str = self.tokenizer.decode(clean_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

            response_str, token_index_map = create_char_to_token_map(self.tokenizer, clean_ids)
            extra_info["token_index_map"] = token_index_map
            extra_info["num_tokens"] = len(clean_ids)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score.pop("score")
                # Store the information including original reward
                for key, value in score.items():
                    if key not in ["score", "acc"]:
                        # for logging purpose
                        key = f"reward_fn/{key}"
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # reward is a *list* over clean_ids; map it back to original
            reward_vec = torch.zeros(valid_response_length, dtype=torch.float32, device=reward_tensor.device)
            for orig_pos, clean_pos in enumerate(id_map):
                if clean_pos != -1:  # not a special token
                    reward_vec[orig_pos] = reward[clean_pos]

            reward_tensor[i, : valid_response_length] = reward_vec
            
            # assert abs(sum(reward) - reward_vec.sum().item()) < 1e-6, f"{sum(reward)=} {reward_vec.sum().item()=}"

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor