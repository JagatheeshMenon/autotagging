import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

UPLOAD_FOLDER = "upload_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Streamlit app
st.title("Generating SEO-friendly tags and keywords from image content")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write(f"Image saved at {image_path}")

    # Open and display the uploaded image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    custom_tag_database = {
        "Wildlife": [
            "lion", "tiger", "elephant", "zebra", "savanna", "jungle", "predator", "prey",
            "monkey", "giraffe", "kangaroo", "penguin", "koala", "crocodile", "snake", "bird"
        ],
        "Landscapes": [
            "mountain", "forest", "ocean", "barn", "beach", "lake", "sunset", "waterfall", "lighthouse",
            "coastal", "desert", "valley", "canyon", "river", "cityscape", "skyscraper",
            "sunset", "sky", "golden hour", "beach", "ocean", "mountain", "nature", "scenic",
            "peaceful", "romantic", "warm", "glowing", "clouds", "silhouette",
            "desert sunset", "beach sunset", "mountain sunset", "city sunset", "rural sunset"
        ],
        "Seasons": [
            "summer", "winter", "spring", "autumn", "snow", "rain", "sunshine", "cloudy", "foggy"
        ],
        "Objects": [
            "car", "bike", "tree", "house", "building", "bridge", "road", "street", "lamp post",
            "keyboard", "laptop", "computer", "typing", "input device", "technology", "gadget", "office", "work",
            "productivity", "key", "qwerty", "wireless", "mechanical", "notebook", "ultrabook", "gaming",
            "macbook", "MacBookPro", "windows", "home office", "remote work", "travel", "coffee shop", "library", "classroom"
        ],
        "Human": [
            "Human standing", "Human walking", "Human running", "Human sitting", "Human smiling",
            "Portrait", "Face", "Eyes", "Smile", "Laughter"
        ],
        "Cities": [
            "New York", "Paris", "Tokyo", "London", "cityscape", "skyline", "metropolis", "urban",
            "Chicago", "Los Angeles", "Sydney", "Melbourne", "Bangkok", "Hong Kong"
        ],
        "Flowers": [
            "Rose", "Flower", "Bloom", "Floral", "Nature", "Petals", "Romantic", "Love", "Garden",
            "SpringFlowers", "FlowersOfTheDay", "FloralBeauty", "Botanical", "RosesAreRed", "FlowerLovers",
            "FragrantFlowers", "FreshBlooms", "RedRoses", "FlowerArrangement", "FloralArt", "Daisy", "Sunflower"
        ],
        "Food": [
            "pizza", "burger", "sushi", "tacos", "ice cream", "cake", "coffee", "tea", "wine",
            "beer", "restaurant", "kitchen", "cooking", "baking", "foodie", "delicious"
        ],
        "Sports": [
            "football", "basketball", "tennis", "baseball", "golf", "soccer", "cricket", "rugby",
            "boxing", "wrestling", "gym", "fitness", "workout", "exercise", "athletics"
        ],
        "Music": [
            "guitar", "piano", "drums", "music", "song", "singer", "band", "concert", "festival",
            "rock", "pop", "jazz", "classical", "hip hop", "rap"
        ],
        "Travel": [
            "airplane", "hotel", "beach", "city", "country", "passport", "suitcase", "backpack",
            "travel", "adventure", "explore", "wanderlust", "tourist", "vacation"
        ]
    }

    # Flatten the tag list (optional: keep category names for later grouping)
    tag_candidates = [tag for category_tags in custom_tag_database.values() for tag in category_tags]

    # Process the image and tag candidates
    inputs = processor(text=tag_candidates, images=image, return_tensors="pt", padding=True)

    # Get similarity scores from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-to-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

    # Get the top-k tags with the highest similarity
    top_k = 5  # Number of top tags to retrieve
    _, top_indices = probs[0].topk(top_k)

    # Retrieve the suggested tags from the candidates
    top_tags = [tag_candidates[i] for i in top_indices]
    st.write("Suggested Tags:", top_tags)

    # Delete option
    if st.button("Delete Image"):
        try:
            os.remove(image_path)
            st.write(f"Image deleted: {image_path}")
        except FileNotFoundError:
            st.write("File not found. It might have already been deleted.")
        except Exception as e:
            st.write(f"Error deleting file: {e}")
else:
    st.write("No file uploaded yet.")
