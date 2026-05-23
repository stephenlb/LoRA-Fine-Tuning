import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def LoRA(model):
    ## TODO ✅ FREEZE Laery no_grad = True?
    ## TODO all layers not just one
    ## TODO train the model "optimization"
    ## TODO optimization trick on output from model N=W+AB ( loading the LoRA learnings into the model )
        ## TODO save old_forward for restoring origin forward
    ## TODO 
    ## get last layer
    rank = 4
    alpha = 16
    layers = []
    lora_layers = []

    for layer in list(model.modules())[-1:]:
        kind = type(layer).__name__
        ## Only work on Linear Layers
        if kind != 'Linear': continue
        print(kind)
        print(layer)
        layers.append(layer)
        shape = layer.weight.shape
        print(shape)
        original_forward = layer.forward
        lora_in = torch.nn.Parameter(torch.rand(shape[1], rank, dtype=torch.bfloat16))
        lora_out = torch.nn.Parameter(torch.zeros(rank, shape[0], dtype=torch.bfloat16))
        lora_layers.append({
            'shape': shape,
            'forward' : original_forward,
            'in' : lora_in,
            'out' : lora_out,
        })
        def forward(x):
            out = original_forward(x)
            lout = x @ lora_in @ lora_out
            return out + lout

        layer.forward = forward

    #layer = layers[-1]
    #shape = layer.weight.shape
    #lora_in = torch.nn.Parameter(torch.rand(shape[1], rank))
    #lora_out = torch.nn.Parameter(torch.zeros(rank, shape[0]))

    return model

accelerator = torch.accelerator.current_accelerator()
print(accelerator)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
#model = model.to(accelerator)
model = LoRA(model)
model.eval()
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
