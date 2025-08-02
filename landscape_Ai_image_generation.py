import os
import requests
import time
import shutil
import random
from openai import OpenAI

# API key for open AI
client = OpenAI(api_key="")

# This is the pathway to the folder containing all of the human made landscape images
real_image_dir = ""

# This is where all the generated images will be saved
output_root = "IMGS"

# This is how many AI-generated images are produced for each real image
ai_images_per_group = 5

# Possible landscape descriptions for the AI prompts
landscape_descriptions = [
    "a foggy mountain valley",
    "a peaceful forest at dawn",
    "a dramatic coastal cliff",
    "a desert with dunes and sky",
    "a snowy alpine lake",
    "a green hillside during spring",
    "a vibrant sunset over fields",
    "a stormy landscape with lightning",
    "a waterfall in a tropical jungle",
    "a winding river through a canyon"
]

# Function that builds a unique prompt for each AI image generation at random
# Combines a random landscape description with the group number
def generate_prompt(filename):
    descriptor = random.choice(landscape_descriptions)
    number = filename.replace("Landscape_", "").replace(".jpg", "").lstrip("0")
    return f"A realistic high-quality landscape photograph of {descriptor}, group {number}"

# This makes a sorted list of all real image filenames
real_images = sorted([f for f in os.listdir(real_image_dir) if f.endswith(".jpg")])

# This is the main loop which loops for each real image
for idx, filename in enumerate(real_images):
    group_num = idx + 1
    
    # This makes paths for each group, one path for real images and one path for AI images
    group_path = os.path.join(output_root, str(group_num))
    real_path = os.path.join(group_path, "REAL")
    ai_path = os.path.join(group_path, "AI")

    # Creates the directories
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(ai_path, exist_ok=True)

    # This makes a copy for the real image into the REAL folder within IMGS with the same filename for each
    src = os.path.join(real_image_dir, filename)
    dst = os.path.join(real_path, "real.jpg")
    shutil.copy2(src, dst)
    print(f"[Group {group_num}] Copied real image: {filename}")

    # This for loop generates multiple AI images for each group
    for i in range(ai_images_per_group):
        prompt = generate_prompt(filename)
        try:
            print(f"[Group {group_num}] Generating AI image {i+1} with prompt: {prompt}")
            
            # This portion uses OpenAI's DALL·E model to create the image
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )

            # This grabs the image URL from the response
            image_url = response.data[0].url

            # Download the image and save it to the corresponding group 'AI' folder
            img_data = requests.get(image_url).content
            save_path = os.path.join(ai_path, f"ai_{i+1}.png")
            with open(save_path, "wb") as f:
                f.write(img_data)

            # Short pause between generations to avoid hitting rate limits
            time.sleep(1)

        except Exception as e:
            print(f"Error generating AI image {i+1} for group {group_num}:\n{e}")