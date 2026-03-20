import os
import json
import numpy as np
from read_emg_closed_v1_new import EMGDataset 
from absl import flags

FLAGS = flags.FLAGS
FLAGS.text_align_directory = "text_alignments"

def extract_emg_features(dataset, output_dir):
    """
    Extract emg_features from EMGDataset and save them as .npy and .json files named according to sentence_index.
    :param dataset: EMGDataset
    :param output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(dataset)):
        sample = dataset[i]  # get single sample
        emg_features = sample['emg'].numpy()  # extract emg_features
        book_location = sample['book_location']  # book location
        text = sample['text']  # target
        sentence_index = book_location[1]  # sentence index
        is_silent = sample['silent']  

        
        if is_silent:
            emg_filename = f"{sentence_index}_silent.npy"
        else:
            emg_filename = f"{sentence_index}_voiced.npy"

        emg_filepath = os.path.join(output_dir, emg_filename)
        json_filepath = os.path.join(output_dir, f"{sentence_index}.json")

        
        np.save(emg_filepath, emg_features)
        print(f"Saved EMG features: {emg_filepath}")

        
        json_data = {
            "book_location": book_location,
            "sentence_index": sentence_index,
            "text": text
        }

        
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"Saved metadata: {json_filepath}")

if __name__ == "__main__":
    import sys
    FLAGS(sys.argv)  

    dataset = EMGDataset(dev=False, test=False, no_testset=True)  
    extract_emg_features(dataset, "extracted_emg_features")  
