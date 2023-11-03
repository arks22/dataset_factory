import cv2
import os

def create_video_from_images(image_directory, output_video_name, frame_rate=30.0):
    """
    Read images from a specified directory and convert them into a video.

    :param image_directory: Path to the directory where images are stored
    :param output_video_name: File name of the output video
    :param frame_rate: Frame rate of the resulting video (default is 30.0)
    """

    # Load the list of png images in the directory
    image_list = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')])

    # If the image list is empty, terminate the process
    if not image_list:
        print("No png images found in the specified directory.")
        return

    # Read the first image to determine the resolution of the video
    image_path = os.path.join(image_directory, image_list[0])
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Set up the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_name, fourcc, frame_rate, (width, height))

    # Convert images to video
    for image_name in image_list:
        image_path = os.path.join(image_directory, image_name)
        image = cv2.imread(image_path)
        video.write(image)

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    print(f"Video output to {output_video_name}.")


# Example usage
create_video_from_images('video_sun', 'output_video.mp4')