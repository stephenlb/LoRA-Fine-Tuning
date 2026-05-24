import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

## LoRA = Low Rank Adaption for Fine Tuning
## Small Matracies Small Matrix  =Small number of Params = FAST

def LoRA(model):
    ## TODO ✅ FREEZE Laery no_grad = True?
    ## TODO ✅ Forward pass with the LoRA layers
    ## TODO ✅ get data
    ## TODO all layers not just one
    ## TODO ✅ train the model "optimization"
    ## TODO ✅ use multiple optimizers
    ## TODO optimization trick on output from model N=W+AB ( loading the LoRA learnings into the model )
        ## TODO save old_forward for restoring origin forward
    ## TODO 
    ## get last layer
    rank = 4
    alpha = 16
    scale = rank / alpha
    layers = []
    lora_layers = torch.nn.ParameterList()

    ## Freeze all layers in LLM
    for param in model.parameters():
        param.requires_grad = False

    for layer in list(model.modules())[-1:]:
        kind = type(layer).__name__
        ## Only work on Linear Layers
        if kind != 'Linear': continue
        print(kind)
        print(layer)
        device = model.device
        layers.append(layer)
        shape = layer.weight.shape
        print(shape)
        original_forward = layer.forward
        lora_in = torch.nn.Parameter(torch.rand(shape[1], rank, dtype=torch.bfloat16)).to(device)
        lora_out = torch.nn.Parameter(torch.zeros(rank, shape[0], dtype=torch.bfloat16)).to(device)
        lora_layers.append(lora_in)
        lora_layers.append(lora_out)
        #lora_layers.append({
        #    'shape': shape,
        #    'forward' : original_forward,
        #    'in' : lora_in,
        #    'out' : lora_out,
        #})
        def forward(x):
            out = original_forward(x)
            lout = (x @ lora_in @ lora_out) * scale
            return out + lout
        layer.forward = forward

    #layer = layers[-1]
    #shape = layer.weight.shape
    #lora_in = torch.nn.Parameter(torch.rand(shape[1], rank))
    #lora_out = torch.nn.Parameter(torch.zeros(rank, shape[0]))

    return model, lora_layers

accelerator = torch.accelerator.current_accelerator()
print(accelerator)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
#model = model.to(accelerator)
model, lora_layers = LoRA(model)
loss = torch.nn.BCELoss()
optimizers = [
    ## May not work with NLLLoss()
    torch.optim.Muon(lora_layers.parameters(), lr=0.001),
    torch.optim.AdamW(lora_layers.parameters(), lr=0.001),
]
#model = model.to(accelerator)

## Training Message and Testing Message
messages = [
    {
      "role": "system",
      "content": "You are a helpful mathematical assistant. When asked to add large numbers, break down the addition step-by-step using place value or column addition, showing your carry-overs, before providing the final sum."
    },
    {
      "role": "user",
      "content": "Calculate the sum of 847,392 and 563,819."
    },
    {
      "role": "assistant",
      "content": "To add 847,392 and 563,819, let's align them by place value and add from right to left:\n\n1. **Ones place:** 2 + 9 = 11 (Write down 1, carry over 1)\n2. **Tens place:** 9 + 1 + 1 (carry) = 11 (Write down 1, carry over 1)\n3. **Hundreds place:** 3 + 8 + 1 (carry) = 12 (Write down 2, carry over 1)\n4. **Thousands place:** 7 + 3 + 1 (carry) = 11 (Write down 1, carry over 1)\n5. **Ten Thousands place:** 4 + 6 + 1 (carry) = 11 (Write down 1, carry over 1)\n6. **Hundred Thousands place:** 8 + 5 + 1 (carry) = 14 (Write down 14)\n\nCombining these results from left to right gives us the final answer.\n\nTherefore, 847,392 + 563,819 = 1,411,211."
    }
]
test_messages = messages[0:2] 

## Train on 100 Samples
for sample in range(100):
    tokenized_outputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    input_ids = tokenized_outputs["input_ids"]
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)

    #for optim in optimizers: optim.zero_grad()
    optim = optimizers[random.randint(0,len(optimizers)-1)]
    optim.zero_grad()
    loss = outputs.loss
    loss.backward()
    optim.step()
    print('loss:', loss)

#outputs.backward()
#print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
#print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
#outputs[0][inputs["input_ids"].shape[-1]:]
#generated_outputs[0][inference_inputs["input_ids"].shape[-1]:]

## Testing
def test():
    inputs = tokenizer.apply_chat_template(
        test_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

test()



#
#generated_tokens = generated_outputs[0][inference_inputs["input_ids"].shape[-1]:]
#print(tokenizer.decode(generated_tokens, skip_special_tokens=True))


#print(outputs)
#print(outputs.shape)
