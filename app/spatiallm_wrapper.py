# app/spatiallm_wrapper.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

class SpatialLMClient:
    def __init__(
        self,
        model_name: str = "manycore-research/SpatialLM1.1-Llama-1B",
        use_gpu: bool = True
    ):
        """
        Loads the SpatialLM model + tokenizer and wraps in a pipeline.
        If use_gpu=True and a CUDA GPU is available, we load in 8-bit on GPU.
        Otherwise we fall back to CPU.
        """
        self.device = 0 if use_gpu else -1
        trust = True  # allow custom model code from HF
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust,
            device_map="auto" if (use_gpu and self.device>=0) else None,
            load_in_8bit=(use_gpu and self.device>=0),
        )
        self.pipe = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def ask(self, scene_description: str, question: str, max_new_tokens: int = 64) -> str:
        """
        Prompt SpatialLM with a scene description + question,
        returns the answer text.
        """
        prompt = f"{scene_description}\n\nQ: {question}\nA:"
        out    = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text   = out[0]["generated_text"]
        # extract everything after "A:"
        return text.split("A:", 1)[-1].strip()
