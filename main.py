import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
from datetime import datetime
from PIL import Image
from manim import *
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, ImageClip
from moviepy.video.fx.all import resize, fadein, fadeout
from moviepy.editor import VideoFileClip, AudioFileClip
from pytube import YouTube , Search

# Function to download and process audio
def download_audio(video_id):
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file_path = audio_stream.download(filename=f"{video_id}.mp3")
    return audio_file_path


def find_and_load_mp4(folder_path):
    if not os.path.exists(folder_path):
        return "The folder does not exist."

    # List all files in the folder
    for file in os.listdir(folder_path):
        # Check if the file has an .mp4 extension
        if file.endswith(".mp4"):
            mp4_path = os.path.join(folder_path, file)
            return mp4_path  # Return the first .mp4 file path
    
    return "No .mp4 file found in the folder."


def add_audio_to_video(mp4_vid_path, audio_file_path):
    # Load the video and audio
    video_clip = VideoFileClip(mp4_vid_path)
    audio_clip = AudioFileClip(audio_file_path)

    # Save the original video without audio
    output_video_no_audio_path = mp4_vid_path.replace(".mp4", "_no_audio.mp4")
    video_clip.set_audio(None).write_videofile(output_video_no_audio_path, codec="libx264")
    
    # Set the audio of the video
    video_with_audio = video_clip.set_audio(audio_clip)

    # Save the final video with the new audio
    output_video_with_audio_path = mp4_vid_path.replace(".mp4", "_with_audio.mp4")
    video_with_audio.write_videofile(output_video_with_audio_path, codec="libx264", audio_codec="aac")
    
    return output_video_no_audio_path, output_video_with_audio_path

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Google Gemini API
if not API_KEY:
    raise ValueError("API key for Google Gemini API is missing. Set it in the .env file.")
genai.configure(api_key=API_KEY)

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
    if os.path.exists(STORIES_FILE):
        with open(STORIES_FILE, "r") as f:
            return json.load(f)
    return []

# Save scenes and images in a timestamped folder
def save_scenes_and_images(scenes, images):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join("project_assets", f"session_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    
    # Save scenes as a text file
    scenes_file = os.path.join(folder_path, "scenes.txt")
    with open(scenes_file, "w") as f:
        for idx, scene in enumerate(scenes):
            f.write(f"Scene {idx + 1}:\n{scene}\n\n")

    # Save images
    image_folder = os.path.join(folder_path, "images")
    os.makedirs(image_folder, exist_ok=True)
    for idx, img in enumerate(images):
        img_path = os.path.join(image_folder, f"scene_{idx + 1}.png")
        img.save(img_path)

    return folder_path

def create_manim_script(images_folder, total_duration):
    image_files = sorted(
        [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(".png")]
    )

    time_per_image = total_duration / len(image_files)

    script_content = f"""
from manim import *

class GeneratedVideo(Scene):
    def construct(self):
        images = {image_files}
        for img in images:
            image = ImageMobject(img)
            image.set_width(config.frame_width)  # Use config.frame_width for proper scaling
            self.add(image)
            self.wait({time_per_image})
            self.remove(image)
"""

    script_path = "generated_video.py"
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    return script_path

# Initialize storage
init_storage()

# Sidebar navigation
st.sidebar.title("Creative Content Generator")
section = st.sidebar.radio("Choose a section:", ["Generate Story", "Saved Stories", "Transform Story", "Create Video", "Audio"])

# Section: Generate Story
if section == "Generate Story":
    st.title("Story Generator")
    story_prompt = st.text_area("Enter a prompt for your story:", "Once upon a time in a magical forest...")
    max_length = st.slider("Story length (in words):", 50, 500, 200)

    if st.button("Generate Story"):
        with st.spinner("Generating story..."):
            try:
                story_generator = pipeline(
                    "text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1
                )
                story = story_generator(
                    story_prompt, max_length=max_length, num_return_sequences=1
                )[0]["generated_text"]
                save_story(story_prompt[:10], story)
                st.success("Story generated successfully!")
                st.subheader("Generated Story")
                st.write(story)
            except Exception as e:
                st.error(f"Error generating story: {e}")

# Section: Saved Stories
elif section == "Saved Stories":
    st.title("Saved Stories")

    try:
        saved_stories = get_stories()
        if saved_stories:
            story_titles = [story["title"] for story in saved_stories]
            selected_story_title = st.selectbox("Select a story:", story_titles)

            selected_story = next(
                (story for story in saved_stories if story["title"] == selected_story_title), None
            )

            if selected_story:
                st.subheader(f"Story: {selected_story['title']}")
                st.write(selected_story["content"])
            else:
                st.error("Error: Story not found. Please select a valid story.")
        else:
            st.info("No stories found. Generate a story to see it here.")
    except Exception as e:
        st.error(f"Error loading stories: {e}")

# Updated Section: Transform Story
elif section == "Transform Story":
    st.title("Transform Story into Scenes")

    try:
        saved_stories = get_stories()
        if saved_stories:
            story_titles = [story["title"] for story in saved_stories]
            selected_story_title = st.selectbox("Select a story:", story_titles)

            selected_story = next(
                (story for story in saved_stories if story["title"] == selected_story_title), None
            )

            if selected_story:
                st.subheader(f"Story: {selected_story['title']}")
                st.write(selected_story["content"])

                min_scenes = st.number_input("Minimum number of scenes:", min_value=1, value=3, step=1)
                min_words_per_scene = st.number_input("Minimum words per scene:", min_value=10, value=30, step=10)

                if st.button("Transform Story"):
                    with st.spinner("Transforming story..."):
                        try:
                            prompt = (
                                f"Break the following story into at max {min_scenes} scenes. "
                                f"Each scene should have at max {min_words_per_scene} words and include detailed descriptions "
                                f"of the setting, characters, and their actions. "
                                f"Here is the story: {selected_story['content']}"
                            )

                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content(prompt)
                            scenes = response.text.split("\n\n")  # Assuming scenes are separated by blank lines
                            st.success("Story transformed successfully!")
                            st.subheader("Transformed Story")
                            for idx, scene in enumerate(scenes):
                                st.markdown(f"### Scene {idx + 1}")
                                st.write(scene)

                            st.header("Generate and Edit Images for Scenes")
                            pipe = StableDiffusionPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
                            ).to("cuda")

                            images_per_scene = st.number_input("Images per scene:", min_value=1, value=1, step=1)
                            generated_images = []

                            for idx, scene in enumerate(scenes):
                                st.write(f"Generating images for Scene {idx + 1}...")
                                for _ in range(images_per_scene):
                                    image = pipe(prompt=scene).images[0]
                                    generated_images.append((idx, image))
                                    st.image(image, caption=f"Scene {idx + 1}")

                            # Allow users to modify images
                            modified_images = []
                            for idx, (scene_idx, image) in enumerate(generated_images):
                                st.write(f"Edit Image for Scene {scene_idx + 1}")
                                uploaded_image = st.file_uploader(
                                    f"Replace Image for Scene {scene_idx + 1}", type=["png", "jpg"], key=f"upload_{idx}"
                                )
                                if uploaded_image:
                                    uploaded_image = Image.open(uploaded_image)
                                    st.image(uploaded_image, caption=f"Updated Scene {scene_idx + 1}")
                                    modified_images.append((scene_idx, uploaded_image))
                                else:
                                    modified_images.append((scene_idx, image))

                            # Save the scenes and modified images
                            if st.button("Save Scenes and Images"):
                                folder_path = save_scenes_and_images(
                                    scenes, [img for _, img in modified_images]
                                )
                                st.success(f"Scenes and images saved in '{folder_path}'.")

                        except Exception as e:
                            st.error(f"Error transforming story or generating images: {e}")
            else:
                st.error("Please select a valid story.")
        else:
            st.info("No stories found. Generate a story to see it here.")
    except Exception as e:
        st.error(f"Error loading stories: {e}")


elif section == "Create Video":
    st.title("Create Video from Saved Assets")
    folder_path = st.text_input("Enter the folder path for the session (e.g., 'project_assets/session_20240101_123456'): ")

    if folder_path and os.path.exists(folder_path):
        images_folder = os.path.join(folder_path, "images")
        if os.path.exists(images_folder):
            image_files = sorted(
                [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(".png")]
            )

            if image_files:
                total_duration = st.number_input("Enter the total duration of the video (in seconds):", min_value=1, value=10, step=1)

                if st.button("Create Video"):
                    with st.spinner("Creating video..."):
                        try:
                            # Calculate duration per image
                            duration_per_image = total_duration / len(image_files)

                            # Create video clip from images
                            from moviepy.editor import ImageSequenceClip

                            video_clip = ImageSequenceClip(image_files, fps=1 / duration_per_image)

                            # Define the output path
                            output_path = os.path.join(folder_path, "story_video.mp4")

                            # Write the video file
                            video_clip.write_videofile(output_path, fps=24, codec="libx264")

                            st.success(f"Video created successfully! Saved at '{output_path}'.")
                            st.video(output_path)

                        except Exception as e:
                            st.error(f"Error creating video: {e}")
            else:
                st.error("No images found in the selected folder.")
        else:
            st.error("Images folder not found in the selected session path.")
    else:
        st.error("Please enter a valid session folder path.")

# Streamlit interface
elif section == "Audio":
    st.title("Add audio to the video generated")
    folder_path = st.text_input("Enter the folder path for the session (e.g., 'project_assets/session_20240101_123456'): ")

    if folder_path and os.path.exists(folder_path):
        mp4_vid = find_and_load_mp4(folder_path)
        
        # Let the user upload an audio file
        uploaded_audio = st.file_uploader("Upload an audio file (.mp3)", type=["mp3"])
        
        if uploaded_audio:
            # Save the uploaded audio file locally
            audio_file_path = os.path.join(folder_path, "uploaded_audio.mp3")
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_audio.read())
            
            # Add the uploaded audio to the video and save both versions
            output_video_no_audio, output_video_with_audio = add_audio_to_video(mp4_vid, audio_file_path)
            
            st.success(f"Audio has been added to the video. The output video with audio is saved as {output_video_with_audio}")
            st.success(f"The original video without audio is saved as {output_video_no_audio}")
