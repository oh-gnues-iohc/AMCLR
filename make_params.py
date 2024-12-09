import tensorflow as tf, tf_keras
from src.amclr.model_tf import AMCLR_TF, AMCLRConfig
from transformers import AutoTokenizer

        
def main():
    config = AMCLRConfig.from_pretrained("google/electra-base-discriminator")
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    special_token_ids = tokenizer.all_special_ids
    
    model = AMCLR_TF(config, special_token_ids)
    model.build()
    model.save_pretrained("./ours_base/")

if __name__ == "__main__":
    main()
