from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import torch

SUMM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    SUMM_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",   # uses GPU if available
)

summarizer_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


def summarize_opinions(comments, max_new_tokens: int = 256) -> str:
    if not comments:
        return "No relevant opinions found."

    # join feedback into one block

    feedback_block = ""
    for c in comments:
        feedback_block += f"- {c.strip()}\n"

    system = (
        "You summarize anonymous feedback about a person named Vasya. "
        "You must be concise, neutral and structured."
    )

    user = f"""
Based on the feedback below, write 4–6 short bullet points.

Rules:
- Do NOT copy sentences from the feedback.
- Do NOT use phrases like "In general" or
  "This is visible both in work and in personal communication".
- Combine similar ideas into one concise point.
- Mention both positive and negative traits if they appear.
- Use neutral, objective tone.
- Output ONLY bullet points, nothing else.

Feedback:
{feedback_block}
"""

    # use Mistral's chat-style prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    out = summarizer_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,        # deterministic, stable
        num_beams=5,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = out[0]["generated_text"]
    # cut off the prompt part if pipeline returns it
    if prompt in generated:
        generated = generated[len(prompt):]

    return generated.strip()


