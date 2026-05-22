import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def LoRA(model):

    ## get last layer
    for layer in list(model.modules())[-1:]:
        last_layer = layer
        break
    shape = last_layer.weight.shape
    rank = 4
    alpha = 16
    lora_in = torch.nn.Parameter(torch.rand(shape[1], rank))
    lora_out = torch.nn.Parameter(torch.zeros(rank, shape[0]))
    def forward(x):
        ## passthrough
        print('it worked!!!!!')
        out = x @ last_layer.weight.T
        print('out')
        print(out)
        return out
    last_layer.forward = forward
    return model

accelerator = torch.accelerator.current_accelerator()
print(accelerator)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = LoRA(model)
messages = [
    {"role": "user", "content": "Who are you?"},
]

#print(model.parameters())

#for layer in list(model.modules())[-1:]:
#    print(layer)
#    print(layer.weight)
#    print(layer.weight.shape)
#

inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
