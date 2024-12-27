import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import vertexai
from vertexai.vision_models import Image as VMImage
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.vision_models import Video as VMVideo
from vertexai.vision_models import VideoSegmentConfig

# Initialize Flask app
app = Flask(__name__)

 
PROJECT_ID = "tutorial-project-437116"  # @param {type:"string"}
LOCATION = "us-central1"


vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load data and model (initialization code to be run only once)

df = pd.read_csv("custom_marks_data.csv")
mm_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")  # Use correct model ID
def get_image_embedding(
    image_path: str = None,
    dimension: int | None = 1408,
) -> list[float]:
    image = VMImage.load_from_file(image_path)
    embedding = mm_embedding_model.get_embeddings(
        image=image,
        dimension=dimension,
    )
    return embedding.image_embedding

def get_text_embedding(
    text: str = "apple muffins",
    dimension: int | None = 1408,
) -> list[float]:
    embedding = mm_embedding_model.get_embeddings(
        contextual_text=text,
        dimension=dimension,
    )
    return embedding.text_embedding

def process_row(row):
    try:
        embedding = get_image_embedding(image_path=row["gcs_path"])
        return embedding
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None

# Function to process text embeddings (corrected)
def process_text_row(row):
    try:
        embedding = get_text_embedding(text=row["combined_text"])
        return embedding
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None
    

# Apply the embedding generation functions
df["image_embeddings"] = df.apply(process_row, axis=1)
df["text_embeddings"] = df.apply(process_text_row, axis=1) 

new_csv_file = "updated_embeddings.csv"
df.to_csv(new_csv_file, index=False)

product_image_list = pd.read_csv("updated_embeddings.csv")
def get_similar_products(query_emb, data_frame):
    # Access the image embeddings
    image_embs = data_frame["image_embeddings"]

    scores = []
    for image_emb in image_embs:
        # Only use eval if image_emb is a string; otherwise, use it directly
        if isinstance(image_emb, str):
            image_emb = eval(image_emb)  # Convert string to list if needed
        scores.append(np.dot(image_emb, query_emb))

    # Add scores to DataFrame and sort by similarity
    data_frame["score"] = scores
    top_results = data_frame.sort_values(by="score", ascending=False).head()
    
    # Return the top results as a dictionary
    return top_results[["score", "combined_text", "img_url"]].to_dict(orient="records")

@app.route("/similar_products", methods=["POST"])
def similar_products():
    text_query = request.json.get("text")
    query_emb = get_text_embedding(text_query)
    results = get_similar_products(query_emb, df)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
