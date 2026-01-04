from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SpatialLMClient:
    def __init__(
        self,
        model_name: str = "manycore-research/SpatialLM1.1-Llama-1B",
        use_gpu: bool = True
    ):
        trust = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        
        if use_gpu:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust,
                device_map="auto",
                load_in_8bit=True,
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust,
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,
            )

    def ask(self, scene_description: str, question: str, max_new_tokens: int = 64) -> str:
        prompt = f"{scene_description}\n\nQ: {question}\nA:"
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = out[0]["generated_text"]
        return text.split("A:", 1)[-1].strip()
