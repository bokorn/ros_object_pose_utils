# Object Pose Utils
A set of tools for object pose estimation with ros.

## Playing a Image Folder:
```
roslaunch object_pose_utils play_image_folder.launch camera_info_file:=/path/to/camera_info.yaml image_folder:=/path/to/image/folder/
```

## Tracking a AR Bundle:
```
roslaunch object_pose_utils image_folder_track_bundle.launch camera_info_file:=/path/to/camera_info.yaml image_folder:=/path/to/image/folder/
```

## Segment Surgical Tools:
Launch this while the AR Bundle Tracking is running. Assumes there is a defined order to the tools. Only tested on IPhone images.
```
roslaunch object_pose_utils image_folder_seg.launch
```
Still a little buggy, dies when it can't look up transform and the bundle stracking is a bit unstable.
