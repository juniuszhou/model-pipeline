import logging


def test_lora_data_loader():
    from transformers import AutoTokenizer

    from mymodel.loader import SFTDataset, chat_template

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.chat_template = chat_template

    dataset = SFTDataset("mymodel/data/lora_identity.jsonl", tokenizer, max_length=128)
    logging.info("dataset: ", dataset[0])


def test_sft_data_loader():
    from transformers import AutoTokenizer

    from mymodel.loader import RLAIFDataset, chat_template

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.chat_template = chat_template

    dataset = RLAIFDataset("mymodel/data/ppo.jsonl", tokenizer, max_length=128)
    logging.info("dataset: ", tokenizer.decode(dataset[0]))
