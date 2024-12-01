import cv2
import mediapipe as mp
import math
from ultralytics import YOLO
import os
import csv


# from gui import draw_text

# All your current analysis code here, wrapped into a function
def analyze_video(main_output_dir="analysis", model_weights="runs/detect/train3/weights/best.pt"):
    """
    Perform video analysis on the specified input video.
    :param input_video_path: Path to the input video file.
    :param main_output_dir: Directory to save analysis output.
    :param model_weights: Path to YOLO model weights.
    :return: Analysis details (output video path, trajectory file path, total shots analyzed).
    """

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Load your trained YOLOv8 model (path to your best-trained weights)
    model = YOLO("runs/detect/train3/weights/best.pt")  # Update this path if the weights are located elsewhere

    # Ensure the main analysis directory exists
    main_output_dir = "analysis"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    # Function to get the next subfolder name
    def get_next_analysis_folder(main_output_dir):
        existing_folders = [f for f in os.listdir(main_output_dir) if
                            f.startswith("analysis") and os.path.isdir(os.path.join(main_output_dir, f))]
        existing_numbers = [int(f.replace("analysis", "")) for f in existing_folders if
                            f.replace("analysis", "").isdigit()]
        next_number = max(existing_numbers) + 1 if existing_numbers else 1
        next_folder = os.path.join(main_output_dir, f"analysis{next_number}")
        if not os.path.exists(next_folder):
            os.makedirs(next_folder)
        return next_folder, next_number

    def get_next_result_filename(subfolder, analysis_number):
        return os.path.join(subfolder, f"results{analysis_number}.txt")

    # Function to get the video filename in the subfolder
    def get_next_video_filename(subfolder, number):
        return os.path.join(subfolder, f"freethrow-analysis-{number}.mp4")

    # Create the next analysis folder and get the video filename
    subfolder, analysis_number = get_next_analysis_folder(main_output_dir)
    output_filename = get_next_video_filename(subfolder, analysis_number)
    output_fps = 30  # Set the desired frame rate for the output video
    output_writer = None

    # Function to initialize a CSV file for storing all trajectories
    def initialize_trajectory_file(subfolder):
        # Extract the folder number from the subfolder name (e.g., "analysis1" -> 1)
        folder_number = ''.join(filter(str.isdigit, os.path.basename(subfolder)))

        # Create the filename with the folder number
        filename = os.path.join(subfolder, f"trajectory_data{folder_number}.csv")

        try:
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Shot Number", "X", "Y"])  # Header
            print(f"Initialized trajectory file at {filename}")
        except Exception as e:
            print(f"Error initializing trajectory file: {e}")

        return filename

    # Function to append trajectory points for a shot to the file
    def append_trajectory_to_file(trajectory_points, shot_number, filename):
        try:
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                for point in trajectory_points:
                    writer.writerow([point[0], point[1], point[2]])  # Append shot number and trajectory
            print(f"Appended shot {shot_number} trajectory to {filename}")
        except Exception as e:
            print(f"Error appending trajectory data: {e}")

    # Initialize the trajectory file for the analysis folder
    trajectory_file = initialize_trajectory_file(subfolder)

    # Function to detect basketball and rim
    def detect_objects(frame, model):
        detected_objects = {"basketball": [], "rim": []}
        results = model(frame)

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, score, class_id = box.cpu().numpy()
                if int(class_id) == 0 and score > 0.3:  # Basketball (class_id = 0)
                    detected_objects["basketball"].append((int(x1), int(y1), int(x2), int(y2), score))
                    print(f"Basketball detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}")
                elif int(class_id) == 1 and score > 0.3:  # Rim (class_id = 1)
                    detected_objects["rim"].append((int(x1), int(y1), int(x2), int(y2), score))
                    print(f"Rim detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}")

        return detected_objects

    # Function to calculate the angle between two points relative to vertical
    def calculate_head_angle(nose, neck):
        dx = neck[0] - nose[0]
        dy = nose[1] - neck[1]  # Positive dy means nose is below neck
        angle_rad = math.atan2(dy, dx)  # Angle in radians
        return math.degrees(angle_rad)  # Convert to degrees

    # Function to calculate the internal knee angle
    def calculate_angle(a, b, c):
        # Calculate vectors
        ab = [a[0] - b[0], a[1] - b[1]]  # Vector from hip to knee
        bc = [c[0] - b[0], c[1] - b[1]]  # Vector from knee to ankle

        # Calculate the dot product and magnitudes
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        # Calculate the angle using arccos of the dot product divided by the magnitudes
        angle_rad = math.acos(dot_product / (mag_ab * mag_bc))
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    # Function to calculate horizontal velocity
    def calculate_horizontal_velocity(prev_x, current_x):
        if prev_x is None:
            return 0
        return abs(current_x - prev_x)

    # Function to detect basketball and rim
    def detect_objects(frame, model):
        detected_objects = {"basketball": [], "rim": []}
        results = model(frame)

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, score, class_id = box.cpu().numpy()
                if int(class_id) == 0 and score > 0.3:  # Basketball (class_id = 0)
                    detected_objects["basketball"].append((int(x1), int(y1), int(x2), int(y2), score))
                    print(f"Basketball detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}")
                elif int(class_id) == 1 and score > 0.3:  # Rim (class_id = 1)
                    detected_objects["rim"].append((int(x1), int(y1), int(x2), int(y2), score))
                    print(f"Rim detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}")

        return detected_objects

    # Load the video
    cap = cv2.VideoCapture('Video/sample.mov')

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer once the video starts processing
    output_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), output_fps,
                                    (frame_width, frame_height))

    # cap = cv2.VideoCapture() - FOR CAMERA FEED
    # Variables to track state
    current_phase = "Dribbling"
    has_shot = False  # Tracks if the player has completed the Shooting phase
    shot_count = 0  # Counts the number of shots taken
    previous_nose_x = None  # To track the nose x-coordinate in the previous frame
    previous_ball_to_wrist_distance = None  # To track the previous ball-to-wrist distance
    min_knee_angle = None  # Tracks the minimum knee angle during each shot
    min_knee_angles = []  # List to store minimum knee angles for all shots
    min_elbow_angle = None  # Tracks the minimum elbow angle during each shot
    min_wrist_angle = None  # Tracks the minimum wrist angle during each shot
    min_elbow_wrist_angles = []  # Store elbow and wrist angles for each shot
    frame_skip = 2  # Frame skipping interval
    frame_count = 0
    trajectory_points = []  # Stores (x, y) positions of the ball
    tracking_ball = False  # Flag to indicate if the ball is being tracked
    # Initialize a list to store shot data
    shot_analysis_data = []

    knee_angles_per_shot = []  # List to store average knee angles for each shot

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Set h and w to the fixed dimensions
        h, w = frame.shape[:2]

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        frame.flags.writeable = True

        # Convert back to BGR for rendering
        image_rgb.flags.writeable = True
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Detect objects
        detected_objects = detect_objects(frame, model)

        # Initialize ball_position
        ball_position = None
        rim_position = None

        # Draw detections on the frame
        for basketball in detected_objects["basketball"]:
            x1, y1, x2, y2, score = basketball
            ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)  # Center of the basketball
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for basketball
            cv2.putText(frame, f"Basketball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for rim in detected_objects["rim"]:
            x1, y1, x2, y2, score = rim
            rim_position = ((x1 + x2) // 2, (y1 + y2) // 2)  # Center of the rim
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for rim
            cv2.putText(frame, f"Rim", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Detect the closest basketball and highest confidence rim
            if len(detected_objects["basketball"]) > 0:
                closest_ball = min(detected_objects["basketball"],
                                   key=lambda b: math.sqrt(
                                       (b[0] - rim_position[0]) ** 2 + (b[1] - rim_position[1]) ** 2))
                ball_position = ((closest_ball[0] + closest_ball[2]) // 2, (closest_ball[1] + closest_ball[3]) // 2)

            if len(detected_objects["rim"]) > 0:
                highest_confidence_rim = max(detected_objects["rim"], key=lambda r: r[4])
                rim_position = ((highest_confidence_rim[0] + highest_confidence_rim[2]) // 2,
                                (highest_confidence_rim[1] + highest_confidence_rim[3]) // 2)

                # Track the ball trajectory
                if ball_position:
                    if current_phase == "Shooting":
                        if not tracking_ball:
                            tracking_ball = True
                            trajectory_points = []  # Start tracking for a new shot
                        trajectory_points.append(
                            (shot_count, ball_position[0], ball_position[1]))  # Include shot number

                    if tracking_ball and rim_position:
                        distance_to_rim = math.sqrt((ball_position[0] - rim_position[0]) ** 2 +
                                                    (ball_position[1] - rim_position[1]) ** 2)
                        if distance_to_rim < 20:  # Ball reaches the rim
                            tracking_ball = False

                        # Save trajectory points to file
                        print(trajectory_points)
                        append_trajectory_to_file(trajectory_points, shot_count, trajectory_file)
                        print(f"Saved trajectory for shot {shot_count}: {trajectory_points}")

                # Draw trajectory lines
                if len(trajectory_points) > 1:
                    for i in range(1, len(trajectory_points)):
                        # Extract (x, y) from trajectory_points
                        prev_point = (
                            trajectory_points[i - 1][1], trajectory_points[i - 1][2])  # (x, y) of the previous point
                        current_point = (
                            trajectory_points[i][1], trajectory_points[i][2])  # (x, y) of the current point
                        cv2.line(frame, prev_point, current_point, (0, 255, 255), 2)  # Yellow line

                # Reset trajectory points for the next shot
                if current_phase == "Ready to Shoot" and not has_shot:
                    trajectory_points = []

            # Show ball position with a dot
            if ball_position:
                cv2.circle(frame, ball_position, 5, (0, 255, 0), -1)  # Green dot at ball position

        # Check if any landmarks were detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of key points
            nose = [int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * h)]
            neck = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)]
            wrist = [int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)]
            ball = [int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * h)]
            shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)]
            hip = [int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                   int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)]
            knee = [int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)]
            ankle = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)]
            elbow = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h)]
            index = [int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * h)]

            # Calculate the head angle
            head_angle = calculate_head_angle(nose, neck)

            # Determine if the wrist is raised
            wrist_raised = wrist[1] < shoulder[1]  # Wrist is above the shoulder

            # Calculate the distance between the ball and wrist
            ball_to_wrist_distance = math.sqrt((ball[0] - wrist[0]) ** 2 + (ball[1] - wrist[1]) ** 2)

            # Calculate horizontal velocity using the nose as a reference
            horizontal_velocity = calculate_horizontal_velocity(previous_nose_x, nose[0])

            # Determine the phase
            if has_shot and horizontal_velocity >= 5:  # Retrieving Ball phase
                current_phase = "Retrieving Ball"
            elif head_angle < -135 and not has_shot:  # Dribbling phase
                current_phase = "Dribbling"
            elif -115 <= head_angle <= -80 and not wrist_raised:  # Getting Ready to Shoot
                current_phase = "Ready to Shoot"
                has_shot = False  # Reset has_shot to track the next shot

                # Calculate the knee angle only during this phase
                knee_angle = calculate_angle(hip, knee, ankle)
                if min_knee_angle is None or knee_angle < min_knee_angle:
                    min_knee_angle = knee_angle  # Track the minimum knee angle

                # Track minimum elbow and wrist angles during shooting phase
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                wrist_angle = calculate_angle(elbow, wrist, index)
                if min_elbow_angle is None or elbow_angle < min_elbow_angle:
                    min_elbow_angle = elbow_angle
                if min_wrist_angle is None or wrist_angle < min_wrist_angle:
                    min_wrist_angle = wrist_angle

                # Draw knee angle visualizations
                cv2.line(frame, tuple(hip), tuple(knee), (255, 255, 0), 2)  # Line from hip to knee
                cv2.line(frame, tuple(knee), tuple(ankle), (255, 255, 0), 2)  # Line from knee to ankle
                cv2.circle(frame, tuple(hip), 5, (0, 255, 0), -1)  # Hip point
                cv2.circle(frame, tuple(knee), 5, (0, 0, 255), -1)  # Knee point
                cv2.circle(frame, tuple(ankle), 5, (255, 0, 0), -1)  # Ankle point

                # Draw knee angle visualization
                cv2.line(frame, tuple(hip), tuple(knee), (255, 255, 0), 2)
                cv2.line(frame, tuple(knee), tuple(ankle), (255, 255, 0), 2)
                # Display the knee angle
                cv2.putText(frame, f'Knee Angle: {int(knee_angle)}', (50, 150),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

                # Draw elbow angle visualization
                cv2.line(frame, tuple(shoulder), tuple(elbow), (0, 255, 255), 2)
                cv2.line(frame, tuple(elbow), tuple(wrist), (0, 255, 255), 2)
                cv2.putText(frame, f'Elbow: {int(elbow_angle)}', (50, 200),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

                # Draw wrist angle visualization
                cv2.line(frame, tuple(elbow), tuple(wrist), (0, 0, 255), 2)
                cv2.line(frame, tuple(wrist), tuple(index), (0, 0, 255), 2)
                cv2.putText(frame, f'Wrist: {int(180 - wrist_angle)}', (50, 250),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)


            elif wrist_raised and previous_ball_to_wrist_distance is not None:
                # Check if the ball leaves the wrist
                distance_increase = ball_to_wrist_distance - previous_ball_to_wrist_distance

                # If the distance increase is greater than 2, it's a shot
                if distance_increase > 2:
                    current_phase = "Shooting"

                    if not has_shot:  # Ensure the shot is counted only once per cycle
                        has_shot = True
                        shot_count += 1  # Increment the shot count

                        # Store the minimum knee, elbow, and wrist angles for this shot
                        knee_form = "Good Form" if min_knee_angle and 99 < min_knee_angle < 118 else "Bad Form"
                        elbow_form = "Good Form" if min_elbow_angle and 66 < min_elbow_angle < 78 else "Bad Form"
                        adjusted_wrist_angle = int(180 - min_wrist_angle) if min_wrist_angle else None
                        wrist_form = "Good Form" if adjusted_wrist_angle and 45 <= adjusted_wrist_angle <= 60 else "Bad Form"

                        shot_analysis_data.append({
                            "shot_number": shot_count,
                            "min_knee_angle": min_knee_angle if min_knee_angle else 0,
                            "knee_form": knee_form,
                            "min_elbow_angle": min_elbow_angle if min_elbow_angle else 0,
                            "elbow_form": elbow_form,
                            "min_wrist_angle": adjusted_wrist_angle if adjusted_wrist_angle else 0,
                            "wrist_form": wrist_form
                        })

                        # Debugging: Print shot data
                        print(f"Shot {shot_count} Data: {shot_analysis_data[-1]}")

                        # Reset angles for the next shot
                        min_knee_angle = None
                        min_elbow_angle = None
                        min_wrist_angle = None

            # Display the shot analysis data on the frame
            y_offset =400  # Starting position for text display
            for shot in shot_analysis_data:
                # Determine the color for the text based on the form
                knee_color = (0, 255, 0) if shot['knee_form'] == "Good Form" else (
                0, 0, 255)  # Green for Good, Red for Bad
                elbow_color = (0, 255, 0) if shot['elbow_form'] == "Good Form" else (0, 0, 255)
                wrist_color = (0, 255, 0) if shot['wrist_form'] == "Good Form" else (0, 0, 255)

                # Create text for shot number
                shot_number_text = f"Shot {shot['shot_number']}:"
                cv2.putText(frame, shot_number_text, (1500, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                            2)  # Black for shot number
                y_offset += 30  # Increment y-offset for the next line

                # Create text for knee angle and form
                knee_text = f"Knee: {int(shot['min_knee_angle'])} ({shot['knee_form']})"
                cv2.putText(frame, knee_text, (1550, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, knee_color, 2)
                y_offset += 30  # Increment y-offset for the next line

                # Create text for elbow angle and form
                elbow_text = f"Elbow: {int(shot['min_elbow_angle'])} ({shot['elbow_form']})"
                cv2.putText(frame, elbow_text, (1550, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, elbow_color, 2)
                y_offset += 30  # Increment y-offset for the next line

                # Create text for wrist angle and form
                wrist_text = f"Wrist: {int(shot['min_wrist_angle'])} ({shot['wrist_form']})"
                cv2.putText(frame, wrist_text, (1550, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, wrist_color, 2)
                y_offset += 40  # Increment y-offset for the next shot

            # Reset the shooting flag once the player transitions back to Dribbling
            if current_phase == "Dribbling" and has_shot:
                has_shot = False

            # Update the previous positions and distances
            previous_nose_x = nose[0]
            previous_ball_to_wrist_distance = ball_to_wrist_distance

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the current phase and shot count
            cv2.putText(frame, f'Phase: {current_phase}', (w - 1900, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'Shots Taken: {shot_count}', (w - 1900, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

            # Display the minimum knee angles for all shots
            y_offset = 200
            for i, (angle, form) in enumerate(min_knee_angles, 1):
                cv2.putText(frame, f'Shot {i}: Min Knee Angle = {int(angle)} ({form})', (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if form == "Good Form" else (0, 0, 255), 2)
                y_offset += 50

            y_offset = 300
            for i, (k_angle, k_form, e_angle, e_form, w_angle, w_form) in enumerate(min_elbow_wrist_angles, 1):
                # Displaying shot number on one line
                cv2.putText(frame, f'Shot {i}:', (50, y_offset),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (209, 51, 245), 2)
                y_offset += 40

                # Displaying knee form on the next line
                cv2.putText(frame, f'Knee Angle = {int(k_angle)} ({k_form})', (50, y_offset),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0) if k_form == "Good Form" else (0, 0, 255), 2)
                y_offset += 40

                # Displaying elbow form on the next line
                cv2.putText(frame, f'Elbow Angle = {int(e_angle)} ({e_form})', (50, y_offset),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0) if e_form == "Good Form" else (0, 0, 255), 2)
                y_offset += 40

                # Displaying wrist form on the next line
                cv2.putText(frame, f'Wrist Angle = {int(180 - w_angle)} ({w_form})', (50, y_offset),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0) if w_form == "Good Form" else (0, 0, 255), 2)
                y_offset += 50

        # Displaying wrist form on the next line
        cv2.putText(frame, f'press q quit', (1550, 1020), cv2.FONT_HERSHEY_COMPLEX, 1.4, (0, 0, 255), 3)

        # Write frame to output video
        output_writer.write(frame)

        # # Show the frame with pose landmarks and annotations
        # cv2.imshow("Basketball Shooting Form", frame)

        # Display the frame
        cv2.imshow("Player Phase Detection", frame)
        print(f"Actual window size: {cv2.getWindowImageRect('Player Phase Detection')}")

        # Exit on 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        save_shot_analysis_to_file(subfolder, analysis_number, shot_analysis_data)

    cap.release()
    output_writer.release()
    cv2.destroyAllWindows()

    print(f"Analysis saved to {output_filename}")

    return output_filename, trajectory_file, shot_count


def calculate_average_angles(text_file_path):
    """
    Calculate the average knee, wrist, and elbow angles dynamically from the shot analysis text file.
    :param text_file_path: Path to the shot analysis text file.
    :return: A dictionary containing the average knee, wrist, and elbow angles.
    """
    try:
        total_knee_angle = 0.0
        total_elbow_angle = 0.0
        total_wrist_angle = 0.0
        shot_count = -1

        with open(text_file_path, "r") as file:
            lines = file.readlines()

        # Debugging: Print lines of the file being processed
        print("Processing the following lines from the text file:")
        for line in lines:
            print(line.strip())

        for line in lines:
            if "Min Knee Angle" in line:
                # Debugging: Print the line being processed
                print(f"Parsing knee angle from line: {line.strip()}")
                knee_angle = float(line.split(":")[1].split("(")[0].strip())
                total_knee_angle += knee_angle
            elif "Min Elbow Angle" in line:
                # Debugging: Print the line being processed
                print(f"Parsing elbow angle from line: {line.strip()}")
                elbow_angle = float(line.split(":")[1].split("(")[0].strip())
                total_elbow_angle += elbow_angle
            elif "Min Wrist Angle" in line:
                # Debugging: Print the line being processed
                print(f"Parsing wrist angle from line: {line.strip()}")
                wrist_angle = float(line.split(":")[1].split("(")[0].strip())
                total_wrist_angle += wrist_angle
            elif line.strip().startswith("Shot"):
                # Increment the shot count and print debug info
                shot_count += 1
                print(f"Shot count incremented: {shot_count}")

        # If no shots were recorded, return None
        if shot_count == 0:
            return {"average_knee_angle": None, "average_elbow_angle": None, "average_wrist_angle": None}

        # Calculate averages
        average_knee_angle = total_knee_angle / shot_count
        average_elbow_angle = total_elbow_angle / shot_count
        average_wrist_angle = total_wrist_angle / shot_count

        # Debugging: Print final calculated averages
        print(f"Final Averages: Knee: {average_knee_angle}, Elbow: {average_elbow_angle}, Wrist: {average_wrist_angle}")
        print(f"Total Knee Angle: {total_knee_angle}")
        print(f"SHOT COUNT: {shot_count}")

        return {
            "average_knee_angle": average_knee_angle,
            "average_elbow_angle": average_elbow_angle,
            "average_wrist_angle": average_wrist_angle,
        }

    except Exception as e:
        print(f"Error calculating averages: {e}")
        return {"average_knee_angle": None, "average_elbow_angle": None, "average_wrist_angle": None}


def save_shot_analysis_to_file(subfolder, analysis_number, shot_data):
    """
    Save the analysis data for each shot to a text file and calculate averages.
    :param subfolder: Directory where the analysis data is saved.
    :param analysis_number: The current analysis number.
    :param shot_data: List of dictionaries containing shot analysis details.
    """
    filename = os.path.join(subfolder, f"shot_analysis_{analysis_number}.txt")
    try:
        with open(filename, mode="w") as file:
            file.write("Shot Analysis Results\n")
            file.write("=======================\n")
            for shot in shot_data:
                file.write(f"Shot {shot['shot_number']}:\n")
                file.write(f"  Min Knee Angle: {shot['min_knee_angle']:.0f} ({shot['knee_form']})\n")
                file.write(f"  Min Elbow Angle: {shot['min_elbow_angle']:.0f} ({shot['elbow_form']})\n")
                file.write(f"  Min Wrist Angle: {shot['min_wrist_angle']:.0f} ({shot['wrist_form']})\n")
                file.write("\n")
            file.write(f"Total Shots: {len(shot_data)}\n")

        # Call calculate_average_angles and append the averages to the file
        averages = calculate_average_angles(filename)
        with open(filename, mode="a") as file:  # Open in append mode
            file.write("\nAverage Angles\n")
            file.write("=======================\n")
            file.write(f"Average Knee Angle: {averages['average_knee_angle']:.0f}\n")
            file.write(f"Average Elbow Angle: {averages['average_elbow_angle']:.0f}\n")
            file.write(f"Average Wrist Angle: {averages['average_wrist_angle']:.0f}\n")

        print(f"Shot analysis and averages saved to {filename}")
    except Exception as e:
        print(f"Error saving shot analysis: {e}")
