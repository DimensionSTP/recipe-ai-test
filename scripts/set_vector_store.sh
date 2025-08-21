#!/bin/bash

path="src/preprocessing"
indices_name="data_base_only.faiss"
items_name="data_base_only.csv"

python $path/set_vector_store.py \
    indices_name=$indices_name \
    items_name=$items_name