from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from transformers import pipeline
import pandas as pd
import torch

class TableRewardFunction:
    def __init__(self, column_info):
        self.column_info = column_info 
        
    def calculate_reward(self, generated_text):
        try:
  
            df = pd.read_csv(pd.compat.StringIO(generated_text))
            reward = 0
            
   
            for col in self.column_info:
                if col['name'] not in df.columns:
                    reward -= 2
                    continue
                
         
                dtype_ok = all(df[col['name']].apply(col['type']))
                reward += 1 if dtype_ok else -1
                
             
                if 'range' in col:
                    reward += df[col['name']].between(*col['range']).mean()
                if 'categories' in col:
                    valid = df[col['name']].isin(col['categories']).mean()
                    reward += valid
                    
            return max(0, reward) 
            
        except Exception as e:
            return -5  


ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("./results")
ppo_tokenizer = T5Tokenizer.from_pretrained('t5-small')

column_info = [
    {'name': 'Age', 'type': int, 'range': (18,60)},
    {'name': 'Gender', 'type': str, 'categories': ['Male','Female']}
]

reward_function = TableRewardFunction(column_info)
sentiment_pipe = pipeline("text-generation", model=ppo_model, tokenizer=ppo_tokenizer)

ppo_config = PPOConfig(
    batch_size=4,
    learning_rate=1e-5,
    ppo_epochs=3,
    remove_unused_columns=False
)

ppo_trainer = PPOTrainer(
    model=ppo_model,
    config=ppo_config,
    tokenizer=ppo_tokenizer
)

for epoch in range(3):
    for batch in dataset["train"]: 
        queries = batch["prompt"]
        
        # Generate responses
        response = sentiment_pipe(
            queries,
            max_length=512,
            num_return_sequences=1,
            return_full_text=False
        )

        rewards = [torch.tensor(reward_function.calculate_reward(r)) 
                  for r in response]
        

        train_stats = ppo_trainer.step(
            queries,
            [r[0]['generated_text'] for r in response],
            rewards
        )

ppo_model.save_pretrained("./rlhf_results")