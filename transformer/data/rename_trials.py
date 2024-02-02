import os

def rename_videos_recursive(root_folder):
    for root, dirs, files in os.walk(root_folder):
        index_counter = 0
        for video_file in files:
            if video_file.lower().endswith('.mp4'):
                folder_name = os.path.basename(root)
                new_name = f'{folder_name.lower()}{index_counter:03d}.mp4'
                index_counter += 1
                old_path = os.path.join(root, video_file)
                new_path = os.path.join(root, new_name)

                # Rename the file
                os.rename(old_path, new_path)
                print(f'Renamed: {video_file} -> {new_name}')

if __name__ == "__main__":
    folder_path = "/Users/florian/Documents/Studium/Master/Semester 2/Study Project/transformer/data/trials"

    if os.path.exists(folder_path):
        rename_videos_recursive(folder_path)
        print("Rename process completed.")
    else:
        print("Folder does not exist. Please provide a valid folder path.")
