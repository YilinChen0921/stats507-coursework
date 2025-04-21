
import os
from PIL import Image
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# Step 1: Load Brazil license plate dataset from Hugging Face
dataset = load_dataset("NeelShah120/brazilian-license-plate-dataset", split="train")

# Step 2: Load pre-trained model & processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Step 3: Preprocessing function
def preprocess(example):
    image = example["image"].convert("RGB")
    text = example["text"]
    encoding = processor(images=image, text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    return {
        "pixel_values": encoding.pixel_values.squeeze(),
        "labels": encoding.labels.squeeze()
    }

# Step 4: Apply preprocessing
dataset = dataset.map(preprocess)

# Step 5: Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-brazil-plate-finetune",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    predict_with_generate=True,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Step 6: Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor
)

# Step 7: Fine-tune
trainer.train()

# Save the model
model.save_pretrained("./trocr-brazil-plate-finetune")
processor.save_pretrained("./trocr-brazil-plate-finetune")
