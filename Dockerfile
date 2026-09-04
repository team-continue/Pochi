FROM nvidia/cuda:12.6.3-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---- basics + locale ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg2 lsb-release locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---- add ROS 2 apt repository (Humble / Ubuntu 22.04) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN ROS_APT_SOURCE_VERSION="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')" && \
    UBUNTU_CODENAME="$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})" && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb

# ---- install ROS 2 Humble ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# コマンドの追加
RUN echo '. /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'export ROS2_WS=ros2_ws' >> /root/.bashrc && \
    echo 'function cw() { cd /${ROS2_WS}; }' >> /root/.bashrc && \
    echo 'function cs() { cd /${ROS2_WS}/src; }' >> /root/.bashrc && \
    echo 'function cb() { cd /${ROS2_WS}; if [ -z $1 ]; then colcon build --symlink-install; else colcon build --symlink-install --packages-select $1; fi; . install/setup.bash && cd -;}' >> /root/.bashrc && \
    echo 'function cbd() { cd /${ROS2_WS}; if [ -z $1 ]; then colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Debug; else colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Debug --packages-select $1; fi; . install/setup.bash && cd -;}' >> /root/.bashrc && \
    echo 'function ws() { if [ $1 ]; then ROS2_WS=$1_ws&&echo "switch ${ROS2_WS}"&&. /${ROS2_WS}/install/setup.bash;fi;}' >> /root/.bashrc && \
    echo "alias cl='cw && rm -rf ./build ./install ./log && cd -'" >> /root/.bashrc && \
    echo '. /ros2_ws/install/setup.bash' >> /root/.bashrc && \
    mkdir -p /ros2_ws/src

# エントリポイント
CMD ["/bin/bash"]