# MovieLens 100K Recommendation System

This repository contains a simple yet powerful recommendation system built on the **MovieLens 100K** dataset. It features both a traditional User-Based Collaborative Filtering (CF) approach and an LLM-powered user profiling script.

> **Note:** The full extracted dataset (`ml-100k/`) and output screenshots were not uploaded to this repository to keep the repository lightweight. A `data.zip` file is included, but you can effortlessly download the full dataset using the provided script.

## Project Highlights

- **User-Based Collaborative Filtering:** Computes user similarities using Cosine Similarity on overlapping items to generate personalized top-N movie recommendations.
- **Explainable Recommendations:** The CF model provides a `reasons` list for each recommendation, showing exactly which similar users contributed to the score and by how much.
- **LLM User Profiling:** Generates concise, natural-language profiles and favorite genre tags for a given user by processing their top ratings using an LLM (Llama 3.1 8B via Ollama). 

## File Structure

- `CF_Cosine.py`: Main script for the Collaborative Filtering model. Recommends top-N movies for a user.
- `LLM_User_Based.py`: Script to generate an LLM-based profile and taste tags for a specific user using Ollama.
- `Data_Download.py`: Utility script to download and extract the MovieLens 100K dataset from GroupLens.
- `utils.py`: Contains data loading, matrix transformations, and similarity math functions.
- `recs_user_196.json`: Sample output of the CF recommendation for User 196.
- `tags_user_196.txt`: Sample output of the LLM profiling for User 196.
- `data.zip`: An archive of the data sample (since the full dataset was not uploaded).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Recommendation_System.git
   cd Recommendation_System
   ```

2. **Install dependencies:**
   Make sure you have `pandas`, `numpy`, and `requests` installed:
   ```bash
   pip install pandas numpy requests
   ```
   *Optional:* For the LLM profiling feature, install `ollama`:
   ```bash
   pip install ollama
   ```
   *(You must also have [Ollama](https://ollama.com/) running locally with the `llama3.1:8b` model pulled.)*

3. **Download the Dataset:**
   Because the full dataset is not included in this repository, run the download script first to fetch the raw `ml-100k` data:
   ```bash
   python Data_Download.py
   ```

## Usage

### 1. Collaborative Filtering Recommendations 
To get Top-N recommendations for a specific user (e.g., User 196) and save them to `outputs/`:
```bash
python CF_Cosine.py --user_id 196 --top_n 5
```

### 2. LLM User Profiling
To generate an AI summary and taste tags for a user based on their viewing history:
```bash
python LLM_User_Based.py --user_id 196
```
*(If Ollama is not installed or running, the script safely falls back to a deterministic summary based on genre counts.)*

---
