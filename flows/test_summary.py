"""Flow to get summaries for a subset of """

from pathlib import Path

from datasets import load_dataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm import tqdm

import random
random.seed(1234)

_root: Path = Path(__file__).parent.parent.resolve()
flow_results: Path = _root / "flow_results"
cleaned_transcripts: Path = flow_results / "cleaned_transcripts"
test_summaries: Path = flow_results / "test_summaries"

if not test_summaries.is_dir():
    test_summaries.mkdir()


filenames = sorted(str(f) for f in cleaned_transcripts.iterdir())
subset_filenames = random.choices(filenames, k=4)

dataset = load_dataset("text", data_files=subset_filenames, sample_by="document")


# LOAD DIFFERENT MODELS FOR TEST:
models = [
    ("philschmid/bart-large-cnn-samsum", 1024, (100, 142)),  # (56, 142)
    ("philschmid/flan-t5-base-samsum", 512, (100, 200)),  # (30, 200)
    # ("MingZhong/DialogLED-base-16384", 16384)
]

def write_doc(contents: list[str], filename: str):
    with open(filename, "w") as f:
        for line in contents:
            f.write(line + "\n")


for model_name, max_length, summary_length in tqdm(models):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    for fname, doc in tqdm(zip(subset_filenames, dataset["train"])):
        text = doc["text"]
        inputs = tokenizer([text], max_length=max_length, truncation=True, return_tensors="pt")

        summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=summary_length[0], max_length=summary_length[1])
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        filename = Path(fname).stem + "_" + model_name.replace("/", "_") + f"({summary_length[0]}-{summary_length[1]})"
        write_doc(summary, test_summaries / f"{filename}.txt")
