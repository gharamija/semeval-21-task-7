import transformers
import torch


class BertEncoder():
    @classmethod
    def get_device(cls):
        if torch.cuda.is_available():
            return torch.device("cuda")
        # if torch.backends.mps.is_built():
        #     return torch.device("mps")
        else:
            return torch.device("cpu")

    def __init__(self):
        model_class, tokenizer_class, pretrained_weights = (
            transformers.BertModel,
            transformers.BertTokenizer,
            "bert-base-uncased",
        )

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_model = model_class.from_pretrained(pretrained_weights)
        self.device = self.get_device()

    def encode(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            padding="max_length",
            max_length=50,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )

        input_id = encoded_dict["input_ids"]
        attention_mask = encoded_dict["attention_mask"]

        input_id, attention_mask = input_id.to(self.device), attention_mask.to(self.device)
        last_hidden_state = self.bert_model(input_id, attention_mask)

        return last_hidden_state[0][:, 0, :].cpu().data.numpy()[0]
