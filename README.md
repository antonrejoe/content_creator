# Transform Story into Scenes with Images and Music

Welcome to the **Transform Story into Scenes** project! This application allows users to transform a written story into a sequence of scenes, generate AI-powered images for each scene, and enhance the experience by adding background music to create a video.

## Features

- **Story Transformation**: Automatically split a story into meaningful scenes with detailed descriptions of settings and characters.
- **AI-Powered Image Generation**: Create visually stunning images for each scene using the Stable Diffusion model.
- **Image Editing**: Replace generated images with custom uploads directly from your local storage.
- **Add Background Music**: Upload a music file to enhance the scenes with a background score.
- **Export Video**: Combine scenes, images, and music into a cohesive video for sharing and playback.

## Tech Stack

- **Frontend**: Streamlit for an interactive and user-friendly interface.
- **Backend**: Generative AI models (Gemini 1.5 Flash for text processing, Stable Diffusion for image generation).
- **Media Handling**: MoviePy for video and audio editing.
- **Python Libraries**:
  - `torch`
  - `Pillow`
  - `moviepy`
  - `streamlit`
  - `transformers`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/transform-story-into-scenes.git
   cd transform-story-into-scenes
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and set up models:
   - Stable Diffusion: Download the model from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5).

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Select a Story**:
   - Choose a saved story or create a new one.
   - Specify the minimum number of scenes and words per scene.

2. **Transform Story**:
   - Click "Transform Story" to generate scenes with detailed descriptions.

3. **Generate Images**:
   - Images are generated automatically for each scene.
   - Replace any image by uploading a new one.

4. **Add Background Music**:
   - Upload a music file in MP3 format.
   - Combine music and scenes into a final video.

5. **Save and Export**:
   - Save all scenes, images, and the final video for future use.

## Folder Structure

```
├── app.py                  # Main application file
├── data/                   # Folder for storing stories and generated content
├── models/                 # Pre-trained AI models
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── utils/                  # Helper functions for story transformation and media handling
```

## Demo

![UI interface](https://github.com/user-attachments/assets/a4a9394d-6de0-481c-bdd1-e9dddc51a730)

## Future Enhancements

- Add support for multiple image generation models.
- Enhance video customization with subtitles and transitions.
- Introduce cloud storage for saved stories and media files.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for hosting pre-trained models.
- [Runway ML](https://runwayml.com/) for Stable Diffusion.
- The Streamlit community for an amazing framework.

---

## Progress 
- Still some critical features have to be added, which are in progress
- Please give me your suggestions and recommendations for this project it would be really helpfull

Feel free to reach out if you have any questions or feedback!
