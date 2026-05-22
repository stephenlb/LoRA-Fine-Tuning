import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LoRA(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.last_layer = list(model.modules())[-1:]
        shape = self.last_layer.weigth.shape
        rank = 4
        self.lora_in = torch.nn.Parameter(torch.rand(shape[1], rank))
        self.lora_out = torch.nn.Parameter(torch.zeros(rank, shape[0]))
        #self.l
        #pass

    def forward(self, x):

#lora = LoRA()
accelerator = torch.accelerator.current_accelerator()
print(accelerator)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
print(model.parameters())
for layer in list(model.modules())[-1:]:
    print(layer)
    print(layer.weight)
    print(layer.weight.shape)

def donothing():
    inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
