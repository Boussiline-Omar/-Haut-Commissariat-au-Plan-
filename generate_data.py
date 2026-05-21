import pandas as pd
import numpy as np
import os
from rgph_pipeline import generate_synthetic_data

if __name__ == "__main__":
    df = generate_synthetic_data(n=100_000)
    df.to_csv("RGPH_projet/Individu.csv", index=False)
    print("RGPH_projet/Individu.csv created.")
