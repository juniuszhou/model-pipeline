def test_data_loader():
    from transformers import AutoTokenizer

    from mymodel.loader import SFTDataset, chat_template

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.chat_template = chat_template

    dataset = SFTDataset("mymodel/data/lora_identity.jsonl", tokenizer, max_length=16)
    print("dataset: ", dataset[0])
