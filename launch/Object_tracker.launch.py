import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription,DeclareLaunchArgument, ExecuteProcess,RegisterEventHandler,OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration,PythonExpression, Command
from launch.event_handlers import OnProcessExit

def launch_setup(context, *args, **kwargs):
    share_dir = get_package_share_directory('depthai_test')
    model_name = LaunchConfiguration('model').perform(context)
    model = os.path.join(share_dir, 'models',model_name)

    camera_node =  Node(
            package='depthai_test',
            executable='depthai_camera_node',
            name='depthai_camera_node',
            output='screen',
            parameters=[{"model_path": model}]
        )

    return [
        camera_node
    ]


def generate_launch_description():

    model_declare =  DeclareLaunchArgument(
            name='model', 
            default_value='mobilenet-ssd_openvino_2021.4_6shave.blob',
            description='Enable use_sime_time to true'
    )
    # sim_declare =  DeclareLaunchArgument(
    #         name='sim', 
    #         default_value='false',
    #         description='Enable use_sime_time to true'
    # )


    return LaunchDescription([
        # sim_declare,
        model_declare,
        OpaqueFunction(function=launch_setup)
    ])