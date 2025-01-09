import streamlit as st
from src.video_processing import process_video
import tempfile

def main():
    st.set_page_config(page_title="Adaptive Surveillance System", layout="wide")
    st.title("Adaptive Surveillance System")

    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        VID_PATH = tfile.name

        with st.spinner('Processing video...'):
            logs, output_video_path = process_video(VID_PATH)

        st.subheader("Detected Moving Objects")
        if logs:
            for log_entry in logs:
                st.write(log_entry)
        else:
            st.write("No significant moving objects detected.")

        st.subheader("Summary Video")
        try:
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
                st.video(video_bytes)
                # Provide a download link
                st.download_button(label="Download Summary Video", data=video_bytes, file_name='summary_video.mp4', mime='video/mp4')
        except Exception as e:
            st.error(f"Error displaying video: {e}")

if __name__ == '__main__':
    main()
