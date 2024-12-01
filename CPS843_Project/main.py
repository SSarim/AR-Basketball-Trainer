from analysis import analyze_video
from gui import display_performance_analysis
import os

if __name__ == "__main__":
# Analyze the video once and pass the results to the GUI
    output_video_path, trajectory_file, total_shots = analyze_video()

    # Construct the path to the shot analysis text file
    subfolder = os.path.dirname(output_video_path)
    analysis_number = os.path.basename(subfolder).replace("analysis", "")
    shot_analysis_file = os.path.join(subfolder, f"shot_analysis_{analysis_number}.txt")

    shot_analysis_file = os.path.join(subfolder, f"shot_analysis_{analysis_number}.txt")
    display_performance_analysis(shot_analysis_file)
