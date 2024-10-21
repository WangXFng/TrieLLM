import json
from transformers import PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LlamaConfig
from trieLogists import Trie, TrieMachine, TrieLogitsProcessor

import argparse


# Loading allowed sequences from a JSON file
def load_allowed_sequences(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def encode_sequences(sequences, tokenizer: PreTrainedTokenizer):
    encoded_sequences = []
    for sequence in sequences:
        token_ids = tokenizer.encode(sequence)
        # encoded_sequences.append(token_ids[1:])
        encoded_sequences.append(token_ids)
    return encoded_sequences


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TrieLLM')
    args = parser.parse_args()
    # args.base_model = "meta-llama/Llama-3.2-1B-Instruct"
    args.base_model = "Qwen/Qwen2.5-1.5B"

    # Loading Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Loading allowed sequences
    allowed_sequences = load_allowed_sequences("allowed_sequences.json")['sequences']

    # Encoding allowed sequences with Tokenizer, return the encoded sequences
    encoded_sequences = encode_sequences(allowed_sequences, tokenizer)
    trie = TrieMachine(tokenizer.eos_token_id, encoded_sequences).getRoot()

    # Custom LogitsProcessor
    num_beams = 2
    logits_processor = LogitsProcessorList([TrieLogitsProcessor(trie, tokenizer, num_beams, ':')])

    # Input prompt
    input_text = "The next token is:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Using Custom LogitsProcessor to control LLM for generations
    # Choice 1. using beam search >>>>>>
    outputs = model.generate(input_ids,
                             logits_processor=logits_processor,
                             max_length=50,
                             num_beams=num_beams,
                             num_return_sequences=num_beams,
                             output_scores=True,
                             return_dict_in_generate=True,
                             early_stopping=True,)

    output_ids = outputs["sequences"]
    scores = outputs["sequences_scores"]
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )
    for output in outputs:
        print(output)

    # # Choice 2. directly generate >>>>>>
    # outputs = model.generate(input_ids,
    #                          logits_processor=logits_processor,
    #                          max_length=50,)
    # 
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(generated_text)
