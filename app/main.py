import sys
sys.path.append('app')  

from spatiallm_wrapper import SpatialLMClient
from scene_processing import build_scene_description_from_pcd

def main():
    #  Point at the SLAM output .ply
    ply_path = "output/room_pointcloud.ply"

    #  Auto-generate the scene description
    scene_desc = build_scene_description_from_pcd(ply_path)
    print(" Scene description:\n", scene_desc)

    #  Start the SpatialLM client
    client = SpatialLMClient(use_gpu=True)

    #  Interactive loop
    print("\n Ready! Ask me about your room:")
    while True:
        q = input(" You: ")
        if q.lower() in ("exit","quit"): break
        ans = client.ask(scene_desc, q)
        print(" Assistant:", ans)

if __name__ == "__main__":
    main()
