import pathlib
from sklearn.cluster import KMeans
import open3d as o3d


def get_dir_paths_to_obj(path_to_dataset, file_obj, include_mdls=[], ):
    path_to_dataset = pathlib.Path(path_to_dataset)
    obj_directories = [dir.name for dir in path_to_dataset.iterdir() if dir.is_dir()]
    
    dirs_with_obj = []
    for o_dir in obj_directories:
        path_to_obj = path_to_dataset / o_dir
        checking_dir = lambda dir: dir.is_dir() and (path_to_obj /dir / file_obj).is_file()
        if include_mdls:
            checking_dir = lambda dir: dir.is_dir() and dir.name in include_mdls and (path_to_obj /dir / file_obj).is_file()

        model_directories = [dir.name for dir in path_to_obj.iterdir() if checking_dir(dir)]
        dirs_with_obj += model_directories
        
        return dirs_with_obj
    
def color_to_label(colors):
    # Extract color information from the point cloud

    # Normalize color values to range [0, 1]
    # colors = colors / 255.0

    # Apply K-means clustering to group similar colors
    n_clusters = 2  # You can adjust this value based on your needs
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(colors)

    # Create a new point cloud with labels

    return labels

def load_pc2tensor(path_to_model_dir, name_file, downsampled_to: int = 0):
    path_to_model = pathlib.Path(path_to_model_dir)
    
    pcd = o3d.io.read_point_cloud(path_to_model)
    
    if downsampled_to > 0:
        pcd = pcd.farthest_point_down_sample(500)
    
    # points = np.asar
    
    
    
    