cmake_minimum_required(VERSION 2.8.3)
project(object_pose_utils)

find_package(catkin REQUIRED
    COMPONENTS 
    message_generation 
    dynamic_reconfigure
    std_msgs 
    tf
    geometry_msgs
    sensor_msgs
    message_filters
    tf2_ros
    cv_bridge
    ar_track_alvar_msgs
    tf2_geometry_msgs
  )

catkin_python_setup()

generate_dynamic_reconfigure_options(
    cfg/ObjectMasker.cfg
)

catkin_package(
    CATKIN_DEPENDS 
    message_runtime 
    dynamic_reconfigure
    std_msgs 
    tf
    geometry_msgs
    sensor_msgs
    message_filters
    image_geometry
    tf2_ros
    cv_bridge
    ar_track_alvar_msgs
    tf2_geometry_msgs
  )

#add_dependencies(pose_labelere ${PROJECT_NAME}_gencfg)

