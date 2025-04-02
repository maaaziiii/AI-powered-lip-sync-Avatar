import gradio as gr
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import subprocess
import os
import shutil


def process(text, video_path, is_static):
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # Generate TTS audio
        tts = gTTS(text=text, lang='en')
        audio_path = "temp/output_audio.wav"
        tts.save(audio_path)

        # Handle static image case
        processed_video_path = video_path  # Default to input video
        if is_static:
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Create video from static image
            processed_video_path = "temp/processed_video.mp4"
            (video_clip
             .set_duration(audio_clip.duration)
             .set_fps(25)
             .write_videofile(processed_video_path))

        # Run Wav2Lip
        output_path = "outputs/result.mp4"
        cmd = f"""
        python inference.py \
        --checkpoint_path checkpoints/wav2lip_gan.pth \
        --face inputs/inputvideo.mp4 \
        --audio inputs/output_audio.wav \
        --outfile "/Users/mazinmurshid/Documents/AS/Wav2Lip/outputs/result.mp4"
        """
        subprocess.run(cmd, shell=True, check=True)
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {str(e)}")
        raise gr.Error(f"Video processing failed: {str(e)}")
        
    except Exception as e:
        print(f"General error: {str(e)}")
        raise gr.Error(f"An error occurred: {str(e)}")
        
    finally:
        # Cleanup temporary files
        if os.path.exists("temp"):
            shutil.rmtree("temp")

# Add this CSS directly in your app.py
CSS = """
/* Embedded CSS Styles */
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
#text_input textarea { 
    font-size: 16px; padding: 12px; min-height: 120px;
    width: 100%; border: 2px solid #e0e0e0; border-radius: 8px; 
}
#upload_box { 
    border: 2px dashed #666; padding: 20px; border-radius: 8px; text-align: center; 
}
#generate_btn { 
    font-size: 18px; padding: 12px 24px; background: #2563eb; 
    color: white; border-radius: 8px; border: none; cursor: pointer; 
}
#output_video { width: 100%; border-radius: 12px; margin-top: 20px; }
.loader { 
    border: 4px solid #f3f3f3; border-top: 4px solid #2563eb; 
    border-radius: 50%; width: 40px; height: 40px; 
    animation: spin 2s linear infinite; margin: 20px auto; 
}
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.progress-text { text-align: center; color: #666; font-size: 14px; }
"""

# Update your Gradio interface setup
with gr.Blocks(css=CSS) as app:

    gr.Markdown("# 🎬 AI Lip Sync Generator")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to Speak",
                placeholder="Enter text to convert to speech...",
                lines=3
            )
            static_check = gr.Checkbox(
                label="Static Image Mode",
                info="Check if using a photo instead of video"
            )
            upload_btn = gr.UploadButton(
                "📁 Upload Video/Image",
                file_types=["video", "image"],
                file_count="single"
            )
            
        output_video = gr.Video(
            label="Generated Video",
            interactive=False,
            format="mp4"
        )
    
    generate_btn = gr.Button("🚀 Generate Video", variant="primary")
    
    generate_btn.click(
        fn=process,
        inputs=[text_input, upload_btn, static_check],
        outputs=output_video
    )

    if __name__ == "__main__":
       app.launch()




   
            