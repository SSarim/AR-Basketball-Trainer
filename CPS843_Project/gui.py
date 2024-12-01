import cv2
import numpy as np
import os
from analysis import analyze_video

def draw_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.5, text_color=(255, 255, 255),
             background_color=(0, 0, 0), thickness=1, padding=10, radius=10):
   # Get text size
   (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
   x, y = position


   # Draw background rectangle
   # Get text size
   (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
   x, y = position


   # Calculate coordinates for the rounded rectangle
   rect_x1, rect_y1 = x - padding, y - text_height - padding
   rect_x2, rect_y2 = x + text_width + padding, y + baseline + padding


   # Draw the rounded rectangle
   cv2.rectangle(frame, (rect_x1 + radius, rect_y1), (rect_x2 - radius, rect_y2), background_color, -1)  # Main body
   cv2.rectangle(frame, (rect_x1, rect_y1 + radius), (rect_x2, rect_y2 - radius), background_color,
                 -1)  # Vertical body
   cv2.circle(frame, (rect_x1 + radius, rect_y1 + radius), radius, background_color, -1)  # Top-left corner
   cv2.circle(frame, (rect_x2 - radius, rect_y1 + radius), radius, background_color, -1)  # Top-right corner
   cv2.circle(frame, (rect_x1 + radius, rect_y2 - radius), radius, background_color, -1)  # Bottom-left corner
   cv2.circle(frame, (rect_x2 - radius, rect_y2 - radius), radius, background_color, -1)  # Bottom-right corner
   # Draw the text
   cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def display_performance_analysis(text_file_path):
    """
    Display the contents of the shot analysis text file.
    :param text_file_path: Path to the shot analysis text file.
    """
    # Create a blank image for displaying text
    img_height, img_width = 1000, 1100
    display_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    display_img[:] = (255, 255, 255)  # White background

    # title text
    draw_text(
        frame=display_img,
        text="Performance Review - Basketball Free Throw Form Analysis",
        position=(50, 100),  # Adjust the position if needed
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        text_color=(1, 1, 1),  # White text
        background_color=(255, 255, 255),  # Blue background
        thickness=3,
        padding=5,
        radius=0  # Rounded corners
    )
    # heading 2: imformation
    draw_text(
        frame=display_img,
        text="Optimal Shooting Angles",
        position=(670, 200),  # Adjust the position if needed
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.8,
        text_color=(255, 0, 0),  # White text
        background_color=(255, 255, 255),  # Blue background
        thickness=2,
        padding=5,
        radius=0  # Rounded corners
    )

    # Indent Body 2: Ideal Knee Angle:    w45-60degrees e roughly 90 w100-118
    draw_text(
        frame=display_img,
        text=f"Ideal Knee Angle: 100 deg to 118 deg",
        position=(627, 230),  # Adjust the position if needed
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.65,
        text_color=(255, 0, 0),  # White text
        background_color=(255, 255, 255),  # Blue background
        thickness=2,
        padding=5,
        radius=0  # Rounded corners
    )

    # Indent Body 2: Ideal Elbow Angle
    draw_text(
        frame=display_img,
        text=f"Ideal Elbow Angle: 90 deg",
        position=(697, 260),  # Adjust the position if needed
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.65,
        text_color=(255, 0, 0),  # White text
        background_color=(255, 255, 255),  # Blue background
        thickness=2,
        padding=5,
        radius=0  # Rounded corners
    )

    # Indent Body 2: Ideal Wrist Angle
    draw_text(
        frame=display_img,
        text=f"Ideal Wrist Angle: 45 deg to 60 deg",
        position=(628, 290),  # Adjust the position if needed
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.65,
        text_color=(255, 0, 0),  # White text
        background_color=(255, 255, 255),  # Blue background
        thickness=2,
        padding=5,
        radius=0  # Rounded corners
    )

    # Drawing "[press 'q' to quit]" on the frame
    draw_text(
        frame=display_img,
        text="[press 'q' to quit]",
        position=(715, 360),  # Adjust position for the bottom of the display
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.7,
        text_color=(255, 255, 255),  # White text
        background_color=(50, 50, 200),  # Blue background
        thickness=2,
        padding=10,
        radius=15  # Rounded corners for prettiness
    )

    # Get the absolute directory path of the text file
    analysis_folder = os.path.abspath(os.path.dirname(text_file_path))

    # Drawing plain text for the absolute folder path
    cv2.putText(
        display_img,
        f"Complete analysis can be accessed at: {analysis_folder}",
        (50, 970),  # Adjust position
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,  # Font scale
        (0, 0, 0),  # Black text color
        1,  # Thickness
        cv2.LINE_AA
    )

    # Read and display the contents of the text file
    try:
        with open(text_file_path, "r") as file:
            lines = file.readlines()

        y_offset = 200  # Starting y-coordinate for text
        for line in lines:
            if y_offset > 960:  # Prevent overflow
                draw_text(
                    frame=display_img,
                    text=line.strip(),
                    position=(50, y_offset),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=0.7,
                    text_color=(0, 0, 0),
                    background_color=(255, 255, 255),
                    thickness=2
                )
                break
            draw_text(
                frame=display_img,
                text=line.strip(),
                position=(70, y_offset),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.65,
                text_color=(0, 0, 0),
                background_color=(255, 255, 255),
                padding=0,
                thickness=2
            )
            y_offset += 30

    except FileNotFoundError:
        draw_text(
            frame=display_img,
            text=f"Error: File {text_file_path} not found!",
            position=(50, 100),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.5,
            text_color=(0, 0, 0),
            background_color=(255, 255, 255),
            thickness=1
        )

    # Automatically save the image in the respective analysis folder
    analysis_folder = os.path.dirname(text_file_path)  # Get the directory of the text file
    save_path = os.path.join(analysis_folder, "performance_analysis.png")  # Save as PNG in the same folder
    cv2.imwrite(save_path, display_img)
    print(f"Performance analysis image saved at: {save_path}")

    # Display the results in a window
    while True:
        cv2.imshow("Shot Analysis Results", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Exit on 'q'
            break

    cv2.destroyAllWindows()