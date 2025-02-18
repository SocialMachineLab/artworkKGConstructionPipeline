import datasets
from datasets import load_dataset
import requests
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ollama
import logging
from collections import defaultdict

# set up logging
logging.basicConfig(
    filename='ner_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_prepare_data():
    """Load CoNLL2003 dataset and prepare for processing."""
    logging.info("Loading CoNLL2003 dataset...")
    
    dataset = load_dataset("conll2003")
    train = dataset['train']
    test = dataset['test']
    valid = dataset['validation']
    
    return train, test, valid

def create_bio_prompt(tokens: List[str]) -> str:
    """Create a prompt for BIO tagging."""
    prompt_bio = """
    You are a precise entity boundary detector. I will provide tokens with their POS tags and chunk information. Your task is to output ONLY B, I, or O labels.

    CRITICAL RULES:
    1. I tag can NEVER appear without a preceding B tag
    2. Every new entity MUST start with B, never with I
    3. If you think a token is part of an entity but has no preceding B, use B instead of I

    Example 1 (CORRECT):
    Manchester    NNP    B-NP
    United    NNP    I-NP
    won    VBD    B-VP
    .    .    O

    Output:
    B
    I
    O
    O

    Example 2 (CORRECT):
    The    DT    B-NP
    New    NNP    I-NP
    York    NNP    I-NP
    Times    NNP    I-NP
    reported    VBD    B-VP

    Output:
    O
    B
    I
    I
    O

    COMMON MISTAKES TO AVOID:
    Wrong:
    [Token]    NNP    B-NP
    [Token]    NNP    I-NP
    Output:
    O    <- Wrong!
    I    <- Wrong! (I cannot appear without B)

    Correct:
    [Token]    NNP    B-NP
    [Token]    NNP    I-NP
    Output:
    B    <- Correct!
    I    <- Correct! (I follows B)

    Entity Start Rules:
    1. If token starts a new entity chunk (B-NP) with NNP -> Usually B
    2. If token is first word of sentence and is NNP -> Consider B
    3. If previous token was O and current is part of entity -> Must use B
    4. Never start an entity with I tag

    For the following input sequence, output ONLY B, I, or O labels, one per line:

    Input:
    {input_text}

    Output:
    """
    return prompt_bio.format(input_text=' '.join(tokens))

def create_type_prompt(tokens: List[str], bio_tags:List[str]) -> str:
    """Create a prompt for entity type prediction."""
    prompt_type= """
    You are a precise entity type classifier. Your task is to output ONLY entity type labels (PER/ORG/LOC/MISC) for marked entities.

    CRITICAL RULES:
    1. Output ONLY the label, one per line
    2. Use ONLY these labels: PER, ORG, LOC, MISC
    3. Only classify tokens marked as entities (B-I sequences)
    4. Non-entity tokens (O) should receive O label

    Entity Types:
    PER - People names
    ORG - Organizations
    LOC - Locations
    MISC - Other named entities

    Input tokens: {tokens}
    BIO tags: {bio_tags}
    Output:
    """
    return prompt_type.format(tokens='\n'.join(tokens), bio_tags='\n'.join(bio_tags))



def predict_entity_types(tokens: List[str], bio_tags: List[str], model: str = "llama3.1:70b") -> List[str]:
    """Predict entity types using OLLAMA"""
    try:
        prompt = create_type_prompt(tokens, bio_tags)
        
        
        response = ollama.generate(
            model=model,
            prompt=prompt
        )
        
        pred_types = response['response'].strip().split('\n')

        pred_types = [tag.strip() for tag in pred_types]
        
        valid_types = {'O', 'PER', 'ORG', 'LOC', 'MISC'}
        if len(pred_types) != len(tokens):
            logging.warning(f"Length mismatch in type prediction: {len(pred_types)} vs {len(tokens)}")
            return None
        
        if not all(tag in valid_types for tag in pred_types):
            logging.warning(f"Invalid entity types found: {pred_types}")
            return None
        
        return pred_types
        
    except Exception as e:
        logging.error(f"Error in entity type prediction: {str(e)}")
        return None

def combine_bio_and_types(bio_tags: List[str], entity_types: List[str]) -> List[str]:
    """Combine BIO and entity types into a single tag."""
    combined = []
    for bio, etype in zip(bio_tags, entity_types):
        if bio == 'O':
            combined.append('O')
        else:
            combined.append(f"{bio}-{etype}")
    return combined

def evaluate_predictions(true_labels: List[List[str]], pred_labels: List[List[str]], 
                        stage: str = "BIO", save_path: str = None):
    """Evaluate predictions and save results."""
    
    true_flat = [item for sublist in true_labels for item in sublist]
    pred_flat = [item for sublist in pred_labels for item in sublist]
    
    
    report = classification_report(true_flat, pred_flat, digits=4)
    conf_matrix = confusion_matrix(true_flat, pred_flat)
    
    
    if save_path:
        with open(f"{save_path}_{stage}_evaluation.txt", 'w') as f:
            f.write(f"Classification Report for {stage}:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {stage}')
    if save_path:
        plt.savefig(f"{save_path}_{stage}_confusion_matrix.png")
    plt.close()
    
    return report, conf_matrix




ner_map = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
}

pos_map = {
    0: '"', 1: "''", 2: "#", 3: "$", 4: "(", 5: ")", 6: ",", 7: ".", 
    8: ":", 9: "``", 10: "CC", 11: "CD", 12: "DT", 13: "EX", 14: "FW", 
    15: "IN", 16: "JJ", 17: "JJR", 18: "JJS", 19: "LS", 20: "MD", 
    21: "NN", 22: "NNP", 23: "NNPS", 24: "NNS", 25: "NN|SYM", 26: "PDT", 
    27: "POS", 28: "PRP", 29: "PRP$", 30: "RB", 31: "RBR", 32: "RBS", 
    33: "RP", 34: "SYM", 35: "TO", 36: "UH", 37: "VB", 38: "VBD", 
    39: "VBG", 40: "VBN", 41: "VBP", 42: "VBZ", 43: "WDT", 44: "WP", 
    45: "WP$", 46: "WRB"
}

chunk_map = {
    0: "O", 1: "B-ADJP", 2: "I-ADJP", 3: "B-ADVP", 4: "I-ADVP",
    5: "B-CONJP", 6: "I-CONJP", 7: "B-INTJ", 8: "I-INTJ", 
    9: "B-LST", 10: "I-LST", 11: "B-NP", 12: "I-NP", 
    13: "B-PP", 14: "I-PP", 15: "B-PRT", 16: "I-PRT", 
    17: "B-SBAR", 18: "I-SBAR", 19: "B-UCP", 20: "I-UCP", 
    21: "B-VP", 22: "I-VP"
}

def convert_to_bio_only(tag: str) -> str:
    """Convert full NER tag to BIO only."""
    if tag == "O":
        return "O"
    return tag[0]  # Returns just B or I


train, test, valid = load_and_prepare_data()


logging.info("Starting BIO prediction...")
processed_data = []

# Predict BIO tags
for idx in tqdm(range(len(test['tokens'])), desc="Processing samples"):
    
    tokens = test['tokens'][idx]
    pos_tags = [pos_map[tag] for tag in test['pos_tags'][idx]]
    chunk_tags = [chunk_map[tag] for tag in test['chunk_tags'][idx]]
    ner_tags = [ner_map[tag] for tag in test['ner_tags'][idx]]
    
    
    input_text = '\n'.join(f"{token}\t{pos}\t{chunk}" 
                            for token, pos, chunk in zip(tokens, pos_tags, chunk_tags))

    
    response = ollama.generate(
        model="llama3.1:70b",
        prompt=create_bio_prompt(input_text)
    )
    pred_bio = response['response'].strip().split("\n")
    
    
    if len(pred_bio) != len(tokens):
        logging.warning(f"Length mismatch in sample {idx}. Expected {len(tokens)}, got {len(pred_bio)}")
        continue
    
    
    cleaned_pred_bio = []
    for tag in pred_bio:
        tag = tag.strip()
        if tag not in ['B', 'I', 'O']:
            if tag.startswith('B-') or tag.startswith('I-'):
                tag = tag[0]
            elif tag == 'O':
                tag = 'O'
        cleaned_pred_bio.append(tag)
    
    
    true_bio = [convert_to_bio_only(tag) for tag in ner_tags]
    
    processed_data.append({
        'tokens': tokens,
        'pos_tags': pos_tags,
        'chunk_tags': chunk_tags,
        'ner_tags': ner_tags,
        'pred_bio': cleaned_pred_bio,
        'true_bio': true_bio
    })


bio_predictions = [data['pred_bio'] for data in processed_data]
true_bio_tags = [data['true_bio'] for data in processed_data]

bio_report, bio_matrix = evaluate_predictions(
    true_bio_tags,
    bio_predictions,
    "BIO",
    "bio"
)
logging.info(f"BIO Stage Results:\n{bio_report}")


logging.info("Starting entity type prediction...")
final_predictions = []
true_ner_tags = []


# Predict entity types
for data in tqdm(processed_data, desc="Predicting entity types"):

    type_pred = predict_entity_types(data['tokens'], data['pred_bio'])
    if type_pred:
        
        combined_pred = combine_bio_and_types(data['pred_bio'], type_pred)
        final_predictions.append(combined_pred)
        true_ner_tags.append(data['ner_tags'])


final_report, final_matrix = evaluate_predictions(
    true_ner_tags,
    final_predictions,
    "Final",
    "final"
)
logging.info(f"Final Stage Results:\n{final_report}")

# Save detailed predictions
with open('detailed_predictions.txt', 'w', encoding='utf-8') as f:
    for i, data in enumerate(processed_data, 1):
        if i <= len(final_predictions):
            f.write(f"Sentence {i}:\n")
            f.write(f"Tokens:     {' '.join(data['tokens'])}\n")
            f.write(f"POS Tags:   {' '.join(data['pos_tags'])}\n")
            f.write(f"Chunk Tags: {' '.join(data['chunk_tags'])}\n")
            f.write(f"NER Tags:   {' '.join(data['ner_tags'])}\n")
            f.write(f"Pred BIO:   {' '.join(data['pred_bio'])}\n")
            f.write(f"True BIO:   {' '.join(data['true_bio'])}\n")
            if i <= len(final_predictions):
                f.write(f"Final Pred: {' '.join(final_predictions[i-1])}\n")
            f.write("\n")


