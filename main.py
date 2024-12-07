import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import json
import os

# Path to the JSON file
STORIES_FILE = "stories.json"

# Initialize the JSON file
def init_storage():
    if not os.path.exists(STORIES_FILE):
        with open(STORIES_FILE, "w") as f:
            json.dump([], f)  # Initialize with an empty list

# Save a story to the JSON file
def save_story(title, content):
    with open(STORIES_FILE, "r") as f:
        stories = json.load(f)
    stories.append({"title": title, "content": content})
    with open(STORIES_FILE, "w") as f:
        json.dump(stories, f)

# Retrieve all stories from the JSON file
def get_stories():
    with open(STORIES_FILE, "r") as f:
        return json.load(f)

# Initialize storage
init_storage()

# App title
st.title("Creative Content Generator")

# # Section: Image Generation
# st.header("Image Generation")

# # Input fields for image generation
# prompt = st.text_input("Enter your image prompt:", "A serene beach at sunset")
# negative_prompt = st.text_input("Enter negative prompt (optional):", "")
# num_images = st.slider("Number of images to generate:", 1, 5, 1)

# if st.button("Generate Images"):
#     with st.spinner("Generating images..."):
#         try:
#             # Load the Stable Diffusion pipeline
#             pipe = StableDiffusionPipeline.from_pretrained(
#                 "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
#             ).to("cuda")
#             generated_images = []
#             for _ in range(num_images):
#                 image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
#                 generated_images.append(image)
            
#             # Display the generated images
#             st.subheader("Generated Images")
#             for idx, image in enumerate(generated_images):
#                 st.image(image, caption=f"Image {idx + 1}", use_container_width=True)
#         except Exception as e:
#             st.error(f"Error generating images: {e}")

# Story generation options
story_prompt = st.text_input("Enter a prompt for your story:", "Once upon a time in a magical forest...")
max_length = st.slider("Story length (in words):", 50, 500, 200)

if st.button("Generate Story"):
    with st.spinner("Generating story..."):
        try:
            # Load text generation pipeline
            story_generator = pipeline(
                "text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1
            )
            story = story_generator(
                story_prompt, max_length=max_length, num_return_sequences=1
            )[0]["generated_text"]
            save_story(story_prompt[:10], story)
            st.subheader("Generated Story")
            st.write(story)

        except Exception as e:
            st.error(f"Error generating story: {e}")

# Section: Saved Stories
st.header("Saved Stories")

try:
    saved_stories = get_stories()
    if saved_stories:
        # List all saved story titles for selection
        story_titles = [story["title"] for story in saved_stories]
        selected_story_title = st.selectbox("Select a story:", story_titles)

        # Check if a story is selected and retrieve the selected story
        selected_story = next(
            (story for story in saved_stories if story["title"] == selected_story_title), None
        )

        if selected_story is None:
            st.error("Error: Story not found. Please select a valid story.")
        else:
            # Display the selected story
            st.subheader(f"Selected Story: {selected_story['title']}")
            st.write(selected_story["content"])

            # Parameters for scene generation
            st.header("Transform Story into Scenes")
            min_scenes = st.number_input("Minimum number of scenes:", min_value=1, value=3, step=1)
            min_words_per_scene = st.number_input("Minimum words per scene:", min_value=20, value=50, step=10)

            if st.button("Transform Story"):
                with st.spinner("Transforming story..."):
                    try:
                        # Load the Flan-T5 model for text-to-text generation
                        transformer = pipeline(
                            "text2text-generation",
                            model="google/flan-t5-small",  # Adjust model size as needed
                            device=0 if torch.cuda.is_available() else -1,
                        )



                        # Generate the transformation prompt
                        prompt = (
                            f"Break the following story into at least {min_scenes} scenes. "
                            f"Each scene should have at least {min_words_per_scene} words and include detailed descriptions of the setting, "
                            f"characters, and their actions. Here is the story: {selected_story['content']}"
                            f"The scenes should be titled as 'Scene 1', 'Scene 2', and so on. "
                            f"The story is {selected_story['content']}"
                        )
                        print(prompt)


                        # Generate the transformed story
                        transformed_story = transformer(
                            prompt, num_return_sequences=1
                        )[0]["generated_text"]
                        # print(transformed_story)
                        # Display the transformed story
                        st.subheader("Transformed Story")
                        st.write(transformed_story)

                    except Exception as e:
                        st.error(f"Error transforming story: {e}")
    else:
        st.info("No stories found. Generate a story to see it here.")
except Exception as e:
    st.error(f"Error loading stories: {e}")
