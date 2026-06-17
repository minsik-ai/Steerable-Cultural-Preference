import warnings
from typing import Union, Dict, Any, Tuple, List, Mapping, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from trl import RewardTrainer


class OurInverseRewardTrainer(RewardTrainer):
    _tag_names = ["trl", "reward-trainer"]


    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
            num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        ref_rewards_chosen = inputs["ref_rewards_chosen"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]
        ref_rewards_rejected = inputs["ref_rewards_rejected"]
        # calculate loss, optionally modulate with margin
        # ratio = 1. + torch.exp(ref_rewards_chosen) / torch.exp(ref_rewards_rejected)
        ratio = torch.maximum(torch.exp(ref_rewards_rejected) / torch.exp(ref_rewards_chosen), torch.ones_like(ref_rewards_chosen))
        # ratio = torch.exp(ref_rewards_chosen) / torch.exp(ref_rewards_rejected)
        # print(ratio)
        if "margin" in inputs:
            loss = -(ratio * nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"])).mean()
        else:
            loss = -(ratio * nn.functional.logsigmoid(rewards_chosen - rewards_rejected)).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
